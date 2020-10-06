cimport cython
from time import perf_counter
from libc.time cimport time_t, time, difftime

import datetime as dt
import os
import csv
import errno
import pickle
from collections import namedtuple


        
@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class ScoreEntry:
    pass

@cython.warn.undeclared(True)
cdef ScoreEntry new_ScoreEntry():
    cdef ScoreEntry entry
    
    entry = ScoreEntry.__new__(ScoreEntry)
    
    return entry

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class Trial:
    def __init__(self, dict args):
        cdef object log_parent_dirname
        cdef object py_save_period
        
        self.system = <BaseSystem?>args["system"] 
        self.domain = <BaseDomain?>args["domain"] 
        self.experiment_name = args["experiment_name"]
        self.mod_name = args["mod_name"]
        
        self.prints_score = args.get("prints_score", False)
        log_parent_dirname = ( 
            args.get(
                "log_parent_dirname", 
                "log"))
        self.deletes_final_save_file = (
            args.get("deletes_final_save_file", True))
        
        # Save period in seconds. 
        py_save_period = args.get("save_period",  dt.timedelta(minutes = 10))
        self.save_period = py_save_period.total_seconds()
        
        self.saves = args.get("saves",  True)
            
            
        self.n_training_episodes_elapsed = 0
        self.n_epochs_elapsed = 0
        self.n_training_steps_elapsed = 0
        self.datetime_str = (
            dt.datetime.now().isoformat()
                .replace("-", "").replace(':', '').replace(".", "_"))
        
        # self.score_history_filename = (
        #     os.path.join(self.log_dirname, "score", "log/%s/%s/score/score_%s.csv" \
        #     % (experiment_name, mod_name, datetime_str)
        self.log_dirname = (
            os.path.join(
                log_parent_dirname, 
                self.experiment_name))
                
        self.score_history = []
        
    cpdef void save(self) except *:
        cdef object save_filename
        cdef object exc
        cdef object save_file
        
        if self.saves:
            save_filename = (
                os.path.join(
                    self.log_dirname, 
                    self.mod_name,
                    "save",
                    "trail_save_{self.datetime_str}.pickle".format(**locals())))
            
            # Create File Directory if it doesn't exist.
            if not os.path.exists(os.path.dirname(save_filename)):
                try:
                    os.makedirs(os.path.dirname(save_filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
                
            with open(save_filename, 'wb') as save_file:
                pickle.dump(self, save_file) 
    
    @staticmethod        
    def load(object save_filename):
        cdef object trial
        cdef object save_file
        
        with open(save_filename, 'rb') as save_file:
            trial = pickle.load(save_file) 
            
        return trial
        
    cpdef void delete_final_save_file(self) except *:
        cdef object save_filename
        
        if self.saves:
            save_filename = (
                os.path.join(
                    self.log_dirname, 
                    self.mod_name,
                    "save",
                    "trail_save_{self.datetime_str}.pickle".format(**locals())))
            
            if os.path.exists(save_filename):    
                os.remove(save_filename) 
        
    cpdef void log_score_history(self) except *:
        cdef object save_filename
        cdef Py_ssize_t entry_id
        cdef object exc
        cdef object save_file
        cdef object writer
        cdef list data
        
        entry_id = 0
        
        save_filename = (
            os.path.join(
                self.log_dirname, 
                self.mod_name,
                "score",
                "score_{self.datetime_str}.csv".format(**locals())))
        
        # Create File Directory if it doesn't exist
        if not os.path.exists(os.path.dirname(save_filename)):
            try:
                os.makedirs(os.path.dirname(save_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        
        data = [0] * len(self.score_history)
        with open(save_filename, 'w', newline='') as save_file:
            writer = csv.writer(save_file)
            
            for entry_id in range(len(self.score_history)):
                data[entry_id] = self.score_history[entry_id].score
            writer.writerow(['score'] + data)
            
            for entry_id in range(len(self.score_history)):
                data[entry_id] = (
                    self.score_history[entry_id].n_epochs_elapsed)
            writer.writerow(['n_epochs_elapsed'] + data)
            
            for entry_id in range(len(self.score_history)):
                data[entry_id] = (
                    self.score_history[entry_id].n_training_episodes_elapsed)
            writer.writerow(['n_training_episodes_elapsed'] + data)
            
            for entry_id in range(len(self.score_history)):
                data[entry_id] = (
                    self.score_history[entry_id].n_training_steps_elapsed)
            writer.writerow(['n_training_steps_elapsed'] + data)

            
    cpdef void run(self) except *:
        cdef BaseSystem system
        cdef BaseDomain domain
        cdef double score
        cdef object observation
        cdef object action
        cdef object feedback
        cdef ScoreEntry score_entry
        cdef time_t current_time
        cdef time_t last_save_time
        
        system = self.system
        domain = self.domain
        
        start = perf_counter()
        
        if self.saves:
            self.save()
        time(&last_save_time)
        
        while not system.is_done_training():

            system.prep_for_epoch()
            domain.prep_for_epoch()
            
            while not system.is_ready_for_evaluation():
                
                domain.reset_for_training()
                
                # Run the domain.
                while not domain.episode_is_done():
                    observation = domain.observation()
                    action = system.action(observation)
                    
                    domain.step(action)
                    
                    feedback = domain.feedback()
                    system.receive_feedback(feedback)
                    
                    self.n_training_steps_elapsed += 1
                
                system.update_policy()
                self.n_training_episodes_elapsed += 1
                
            # End training; Begin evaluation.
            domain.reset_for_evaluation()
    
            # Run the domain.
            while not domain.episode_is_done():
                observation = domain.observation()
                action = system.action(observation)
                
                domain.step(action)
                    
            score = domain.score()
            system.receive_score(score)
            
            # End of evaluation.
            
            self.n_epochs_elapsed += 1
            
            score_entry = new_ScoreEntry()
            score_entry.score = score
            score_entry.n_epochs_elapsed = self.n_epochs_elapsed
            score_entry.n_training_episodes_elapsed = (
                self.n_training_episodes_elapsed)
            score_entry.n_training_steps_elapsed = self.n_training_steps_elapsed
            
            self.score_history.append(score_entry)
                
            if self.prints_score:
                # Print last score entry.
                print(
                    "score: {score_entry.score}\n"
                    "n_epochs_elapsed: "
                    "{score_entry.n_epochs_elapsed}\n "
                    "n_training_episodes_elapsed: "
                    "{score_entry.n_training_episodes_elapsed}\n"
                    "n_training_steps_elapsed: "
                    "{score_entry.n_training_steps_elapsed}\n"
                    .format(**locals()))
            
            time(&current_time)
            if difftime(current_time, last_save_time) > self.save_period:
                if self.saves:
                    self.save()
                time(&last_save_time)
            end = perf_counter()
            
            print(end-start)
            start = end
        

        self.log_score_history()
        system.output_final_log(
            os.path.join(
                self.log_dirname, 
                self.mod_name),
            self.datetime_str)
        domain.output_final_log(
            os.path.join(
                self.log_dirname, 
                self.mod_name),
            self.datetime_str)
        
        if self.deletes_final_save_file and self.saves:
            self.delete_final_save_file()
        
        