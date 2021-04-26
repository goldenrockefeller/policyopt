from time import perf_counter

import datetime as dt
import os
import csv
import errno
import pickle
from collections import namedtuple



class ScoreEntry:
    pass

class Trial:
    def __init__(self, args):
        self.system = args["system"]
        self.domain = args["domain"]
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

    def save(self):

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
    def load(save_filename):

        with open(save_filename, 'rb') as save_file:
            trial = pickle.load(save_file)

        return trial

    def delete_final_save_file(self):

        if self.saves:
            save_filename = (
                os.path.join(
                    self.log_dirname,
                    self.mod_name,
                    "save",
                    "trail_save_{self.datetime_str}.pickle".format(**locals())))

            if os.path.exists(save_filename):
                os.remove(save_filename)

    def log_score_history(self):
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


    def run(self):
        system = self.system
        domain = self.domain

        start = perf_counter()



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

            score_entry = ScoreEntry()
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

