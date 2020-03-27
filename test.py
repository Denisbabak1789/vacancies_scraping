import json
import os
import subprocess
import tarfile
import configparser
import requests
import numpy as np
import shutil

import settings
from be_sync_module.be_syncer import BESyncerSync
from models import ModelFile, Module, db
from periodic_scripts.runner import SyncerTaskRunner, periodic_task


class ModelUpdater(SyncerTaskRunner):
    def __init__(self):
        super(ModelUpdater, self).__init__()
        self._model_dir = settings.MODELS_DIR  # read from settings.py
        os.makedirs(self._model_dir, exist_ok=True)
        self._sync_syncer = BESyncerSync()
        self.config_parser = configparser.ConfigParser(strict=False)

    #        self.update_current_model_version()

    def _update_current_model_version(self, default_model_id=0):
        current_model = ModelFile.select(ModelFile.model_id).where(ModelFile.active_status == True)
        try:
            current_model_id = current_model[0].model_id
            box = Module.get()
            box_id = box.server_id
            self._send_current_version(current_model_id, box_id)
        except Exception as e:  # no current version
            box = Module.get()
            box_id = box.server_id
            self._send_current_version(default_model_id, box_id)  # default model id
            pass

    def _send_current_version(self, current_model_id, box_id):
        return self._sync_syncer._post("/api/v1/models/update_version",
                                       data={"model_version_id": current_model_id, "box_id": box_id, "is_applied": True})

    @periodic_task(settings.UPDATE_MODEL_INTERVAL, 'Exception in model updating loop')
    async def model_updater_loop(self):
        box = Module.get()
        model_current_version = box.model_current_version
        model_new_version = box.model_new_version
        if model_new_version:
            if model_new_version['name'] != model_current_version['name']:
                self.logger.info("log_type = event, component = ml_model_processing, event_type = update_model, "
                                 "message = 'Model should has been downloaded and applied in core', "
                                 "model_new_version = {}, model_current_version = {}, severity = high",
                                 model_new_version['name'], model_current_version['name'])
                if model_new_version['name'] == 'default':
                    status_updater = ModelFile.update({ModelFile.active_status: False})
                    status_updater.execute()
                    self._update_current_model_version(model_new_version['id'])
                    update_new_model_version_field = Module.update({Module.model_new_version_json: "null" })
                    update_new_model_version_field.execute()
                    update_current_model_version_field = Module.update({Module.model_current_version_json: json.dumps(model_new_version) })
                    update_current_model_version_field.execute()
                    self.logger.info("log_type = event, component = ml_model_processing, event_type = update_model, "
                                 "message = 'Core will be restarted to apply the default version', "
                                 "model_new_version = {}, model_current_version = {}, severity = high",
                                 model_new_version['name'], model_current_version['name'])
                    subprocess.Popen(['bash', './kill_all.sh'])  # restart

                else:
                    try:
                        respond_for_model_url = self._sync_syncer._get(
                            "/api/v1/models/{}".format(model_new_version['id']))
                    except Exception as e:
                        self.logger.warning("can't load new version from backend, updating terminated_{}", e)
                        return
                    model_meta_data = json.loads(respond_for_model_url)
                    if model_meta_data['meta']:
                        self._download_extract_file(model_new_version, model_meta_data)

    def _check_model_existance(self, model_new_version, model_meta_data):  # could be file attributes to check
        if os.path.exists("{}/{}/{}".format(self._model_dir, model_new_version['name'],
                                            model_meta_data['archive_name'].split('/')[-1])):
            return True
        else:
            return False

    def _extract_archive(self, model_name, archive_name):
        tar_file = tarfile.open("{}/{}/{}".format(self._model_dir, model_name, archive_name))
        tar_file.extractall("{}/{}".format(self._model_dir, model_name))  # check files
        tar_file.close()

    def _download_extract_file(self, model_new_version, model_meta_data):
        try:
            os.mkdir("{}/{}".format(self._model_dir, model_new_version['name']))
        except:
            pass

        model_name = model_new_version['name']
        archive_name = model_meta_data['archive_name'].split('/')[-1]
        model_existance = self._check_model_existance(model_new_version, model_meta_data)
        self.download_status = False

        if model_existance:
            self.download_status = True
        else:
            try:
                with requests.get(model_meta_data['presign_url'], stream=True, timeout=15) as r:
                    r.raise_for_status()
                    with open("{}/{}/{}".format(self._model_dir, model_name, archive_name), 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                self._extract_archive(model_name, archive_name)
                yolo_status = self.check_yolo_model(
                    "{}/{}/{}".format(self._model_dir, model_name, model_meta_data['meta']['yolo_path']))
                classifier_status = self.check_classifier_model(
                    "{}/{}/{}".format(self._model_dir, model_name, model_meta_data['meta']['classifier_path']))
                if yolo_status and classifier_status:
                    self.download_status = True
                else:
                    shutil.rmtree("{}/{}".format(self._model_dir, model_name))  # remove model foler
            except Exception as e:  # download error or broken archive
                self.logger.error("can't load data, error = {}", e)

        if self.download_status:
            if len(ModelFile.select().where(ModelFile.model_name == model_name)) == 0:
                self._save_model_name_in_db(model_new_version['id'], model_name, model_meta_data['presign_url'],
                                            json.dumps(model_meta_data['meta']))

            self._update_model_in_db(model_new_version['id'], model_name, model_meta_data['presign_url'],
                                     json.dumps(model_meta_data['meta']))
            self._set_active_model(model_name)

    def _save_model_name_in_db(self, model_id, model_name, s3_url, model_dir_structure):
        model = ModelFile(
            model_id=model_id,
            model_name=model_name,
            s3_link=s3_url,
            active_status=False,
            model_dir_structure=model_dir_structure
        )
        model.save()

    def _update_model_in_db(self, model_id, model_name, s3_url, model_dir_structure):
        model_updater_query = ModelFile.update({
            ModelFile.model_id: model_id,
            ModelFile.s3_link: s3_url,
            ModelFile.model_dir_structure: model_dir_structure}
        ).where(
            ModelFile.model_name == model_name
        )
        model_updater_query.execute()

    def _set_active_model(self, model_name):
        with db.atomic() as transaction:  # Opens new transaction.
            try:
                status_updater = ModelFile.update({ModelFile.active_status: False})
                status_updater.execute()
                status_activation = ModelFile.update({ModelFile.active_status: True}).where(
                    ModelFile.model_name == model_name
                )
                status_activation.execute()
            except Exception as e:
                transaction.rollback()
            else:
                self._update_current_model_version()
                box = Module.get()
                model_current_version = box.model_current_version
                update_new_model_version_field = Module.update({Module.model_new_version_json: "null" })
                update_new_model_version_field.execute()
                current_model = ModelFile.select(ModelFile.model_id).where(ModelFile.active_status == True)
                current_model_id = current_model[0].model_id
                current_model = ModelFile.select(ModelFile.model_name).where(ModelFile.active_status == True)
                current_model_name = current_model[0].model_name
                update_current_model_version_field = Module.update({Module.model_current_version_json: json.dumps({"id": current_model_id, "name": current_model_name}) })
                update_current_model_version_field.execute()
                self.logger.info("log_type = event, component = ml_model_processing, event_type = update_model, "
                                 "message = 'Core will be restarted to apply a new model version', " 
                                 "model_new_version = {}, model_current_version = {}, severity = high",
                                 current_model_name, model_current_version['name'])
                subprocess.Popen(['bash', './kill_all.sh'])  # restart

    def check_file_existance(self, file_path):
        return os.path.exists(file_path)

    def check_yolo_model(self, model_path):
        model_files = ['model.cfg', 'model.data', 'model.names', 'model.weights', 'model_class.names']
        for file in model_files:  # check file existance
            if not self.check_file_existance("{}/{}".format(model_path, file)):
                return False
        with open("{}/model.cfg".format(model_path)) as open_file:
            read_model_cfg = open_file.read()
            try:
                self.config_parser.read_string(read_model_cfg)
                width = int(self.config_parser['net']['width'])
                height = int(self.config_parser['net']['height'])  # could be other verifications
            except Exception as e:
                return False
        with open("{}/model.data".format(model_path)) as open_file:
            read_model_data = open_file.readlines()
            if len(read_model_data) == 0:  # some verification
                return False
        with open("{}/model.names".format(model_path)) as open_file:
            read_model_names = open_file.readlines()
            if len(read_model_names) == 0:  # some verification
                return False
        try:
            weights = np.fromfile("{}/model.weights".format(model_path), dtype=float)
            del weights
        except Exception as e:
            return False
        with open("{}/model_class.names".format(model_path)) as open_file:
            try:
                read_model_class_names = json.load(open_file)
            except Exception as e:
                return False
        return True

    def check_classifier_model(self, model_path):
        return any(['.pt' in file_name for file_name in os.listdir(model_path)])


if __name__ == '__main__':
    ModelUpdater().run_tasks()
