cimport cython

import datetime as dt
import os
import csv
import errno
import pickle
from collections import namedtuple



ScoreEntry = (
    namedtuple(
        "ScoreEntry", 
        [
        "score",
        "n_generations_elapsed",
        "n_training_episodes_elapsed",
        "n_training_steps_elapsed"
        ]))


@cython.warn.undeclared(True)
cdef class Trial:
    def __init__(self, args):
        self.system = <BaseSystem?>args["system"] 
        self.domain = <BaseDomain?>args["domain"] 
        self.experiment_name = args["experiment_name"]
        self.mod_name = args["mod_name"]
        
        self.prints_score = args.get("prints_score", default = False)
        log_parent_dirname = args.get("log_parent_dirname", default = "log")
        self.deletes_final_save_file = (
            args.get("deletes_final_save_file", default = True))
        
        # Save period in seconds. 
        self.save_period = (
            args.get("save_period", default = dt.timedelta(minutes = 10)))
            
        self.n_training_episodes_elapsed = 0
        self.n_generations_elapsed = 0
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
                self.experiment_name, 
                self.mod_name))
                
        self.score_history = []
        
    cpdef void save(self) except *:
        cdef object save_filename
        
        save_filename = (
            os.path.join(
                self.log_dirname, 
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
    def object load(object save_filename):
        cdef object trial
        
        with open(save_filename, 'rb') as save_file:
            trial = pickle.load(self, save_file) 
            
        return trial
        
    cpdef void delete_final_save_file(self) except *:
        cdef object save_filename
        
        save_filename = (
            os.path.join(
                self.log_dirname, 
                "save",
                "save_{self.datetime_str}.pickle".format(**locals())))
        
        if os.path.exists(save_filename):    
            os.remove(save_filename) 
        
    cpdef void log_score_history(self) except *:
        cdef object save_filename
        cdef Py_ssize_t
        
        save_filename = (
            os.path.join(
                self.log_dirname, 
                "score",
                "score_{self.datetime_str}.csv".format(**locals())))
        
        # Create File Directory if it doesn't exist
        if not os.path.exists(os.path.dirname(save_filename)):
            try:
                os.makedirs(os.path.dirname(save_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
                    
        with open(save_filename, 'w', newline='') as save_file:
            writer = csv.writer(save_file)
            writer.writerow(
                ['score'] 
                + [
                self.score_history[entry_id].score
                for entry_id 
                in range(len(self.score_history))
                ])
            writer.writerow(
                ['n_generations_elapsed'] 
                + [
                self.score_history[entry_id].n_generations_elapsed
                for entry_id 
                in range(len(self.score_history))
                ])
            writer.writerow(
                ['n_training_episodes_elapsed'] 
                + [
                self.score_history[entry_id].n_training_episodes_elapsed
                for entry_id 
                in range(len(self.score_history))
                ])
            writer.writerow(
                ['n_training_steps_elapsed'] 
                + [
                self.score_history[entry_id].n_training_steps_elapsed
                for entry_id 
                in range(len(self.score_history))
                ])
            

            
    cpdef void run(self) except *:
        cdef BaseSystem system
        cdef BaseDomain domain
        cdef double score
        cdef object observations
        cdef object actions
        cdef object feedback
        cdef object last_save_datetime
        
        system = self.system
        domain = self.domain
        
        last_save_datetime = None
        
        while not system.is_done_training():

            system.prep_for_generation()
            domain.prep_for_generation()
            
            while not system.is_ready_for_evalution():
                
                domain.reset_for_training()
                
                # Run the domain.
                while not domain.episode_is_done():
                    observations = domain.observations()
                    actions = system.actions(observations)
                    
                    domain.step(actions)
                    
                    feedback = domain.feedback()
                    system.receive_feedback(feedback)
                    
                    self.n_training_steps_elapsed += 1
                
                system.update_policy()
                self.n_training_episodes_elapsed += 1
                
            # End training; Begin evalution.
            domain.reset_for_evaluation()
    
            # Run the domain.
            while not domain.episode_is_done():
                observations = domain.observations()
                actions = system.actions(observations)
                
                domain.step(actions)
                    
            score = domain.score()
            system.receive_score(score)
            
            # End of evalution.
            
            self.n_generations_elapsed += 1
            
            self.score_history.append(
                ScoreEntry(
                    score = score,
                    n_generations_elapsed = self.n_generations_elapsed,
                    n_training_episodes_elapsed = (
                        self.n_training_episodes_elapsed),
                    n_training_steps_elapsed = self.n_training_steps_elapsed))
            
            if prints_score:
                # Print last score entry.
                print(self.score_history[-1])
                
            if last_save_datetime is None:
                self.save()
                last_save_datetime = dt.datetime.now()
            elif dt.datetime.now() - last_save_datetime > self.save_period:
                self.save()
                last_save_datetime = dt.datetime.now()
        

        self.log_score_history()
        system.output_final_log(self.log_dirname, self.datetime_str)
        domain.output_final_log(self.log_dirname, self.datetime_str)
        
        if self.deletes_final_save_file:
            self.delete_final_save_file()
        
        