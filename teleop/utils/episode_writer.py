import os
import cv2
import json
import datetime
import numpy as np
import time
from copy import deepcopy
from .rerun_visualizer import RerunLogger
from queue import Queue, Empty
from .recording_schema import BarcodeReceiver
from threading import Lock, Thread
import logging_mp
import re
import shutil
logger_mp = logging_mp.getLogger(__name__)

def beep_scan():
    try:
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        client = AudioClient()
        client.SetTimeout(2.0)
        client.Init()
        t = np.arange(1920) / 16000
        env = np.clip(np.minimum(t, 0.12 - t) / 0.005, 0, 1)
        pcm = (10000 * env * np.sin(2 * np.pi * 1600 * t)).astype("<i2").tobytes()
        rc, _ = client.PlayStream("scan", str(time.monotonic_ns()), pcm)
        if rc:
            logger_mp.warning(f"Scan beep returned {rc}")
    except Exception as exc:
        logger_mp.warning(f"Scan beep failed: {exc}")


class EpisodeWriter():
    def __init__(self, task_dir, task_goal=None, task_desc = None, task_steps = None,
                 frequency=30, image_size=[640, 480], rerun_log = True,
                 extra_info=None, voice_prompts=False, barcode_scanner=False, scan_beep=False):
        """
        image_size: [width, height]
        """
        logger_mp.info("==> EpisodeWriter initializing...")
        self.task_dir = task_dir
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "task description",
            "steps":"step1: do this; step2: do that; ...",
        }
        if task_goal is not None:
            self.text['goal'] = task_goal
        if task_desc is not None:
            self.text['desc'] = task_desc
        if task_steps is not None:
            self.text['steps'] = task_steps

        self.frequency = frequency
        self.image_size = image_size

        self.rerun_log = rerun_log
        if self.rerun_log:
            logger_mp.info("==> RerunLogger initializing...")
            self.rerun_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit = "300MB")
            logger_mp.info("==> RerunLogger initializing ok.")
        
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_ids = []
            for entry in os.listdir(self.task_dir):
                match = re.fullmatch(r"episode_(\d+)", entry)
                if match and os.path.isdir(os.path.join(self.task_dir, entry)):
                    episode_ids.append(int(match.group(1)))
            self.episode_id = max(episode_ids, default=-1)
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info(f"==> episode directory does not exist, now create one.")
        self.data_info(extra_info=extra_info)
        self.episode_start_monotonic_ns = None

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.worker_thread = Thread(target=self.process_queue)
        # Author: mingyang - Queue recording prompts so audio RPCs never block the control loop.
        self.voice_queue = Queue() if voice_prompts else None
        if voice_prompts:
            self.voice_thread = Thread(target=self._play_prompts, name="recording-voice", daemon=True)
            self.voice_thread.start()
        notify = (lambda: Thread(target=beep_scan, daemon=True).start()) if scan_beep else None
        self.scanner = BarcodeReceiver(on_scan=notify) if barcode_scanner else None
        self.scan_summary = None
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.")
    
    def _play_prompts(self):
        client = None
        while True:
            message = self.voice_queue.get()
            if message is None:
                return
            try:
                if client is None:
                    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
                    client = AudioClient()
                    client.SetTimeout(2.0)
                    client.Init()
                    code = client.SetVolume(60)
                    if code != 0:
                        raise RuntimeError(f"SetVolume returned {code}")
                code = client.TtsMaker(message, 0)
                if code != 0:
                    raise RuntimeError(f"TtsMaker returned {code}")
            except Exception as exc:
                logger_mp.warning(f"Recording voice prompt failed: {exc}")
                client = None

    def is_ready(self):
        return self.is_available

    def data_info(self, version='1.0.0', date=None, author=None, extra_info=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "unitree" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   [],
                    "left_ee":  [],
                    "right_arm":  [],
                    "right_ee": [],
                    "body":       [],
                },

                "tactile_names": {
                    "left_ee": [],
                    "right_ee": [],
                }, 
                "sim_state": ""
            }
        if extra_info:
            self.info.update(deepcopy(extra_info))

 
    def create_episode(self, episode_id: int = None):
        """
        Create a new episode.
        Args:
            episode_id (int, optional): The starting item ID for the episode. Defaults to None
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        candidate_id = self.episode_id + 1 if episode_id is None else int(episode_id)
        episode_dir = os.path.join(self.task_dir, f"episode_{str(candidate_id).zfill(4)}")
        if os.path.exists(episode_dir):
            logger_mp.error(f"==> Refusing to overwrite existing episode: {episode_dir}")
            return False

        # Creating the episode directory without exist_ok makes allocation
        # collision-safe even if another writer races for the same ID.
        try:
            os.makedirs(episode_dir, exist_ok=False)
        except FileExistsError:
            logger_mp.error(f"==> Episode allocation collision, recording not started: {episode_dir}")
            return False

        self.episode_id = candidate_id
        self.episode_dir = episode_dir
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write('{\n')
            f.write('"info": ' + json.dumps(self.info, ensure_ascii=False, indent=4) + ',\n')
            f.write('"text": ' + json.dumps(self.text, ensure_ascii=False, indent=4) + ',\n')
            f.write('"data": [\n')
        self.first_item = True   # Flag to handle commas in JSON array
        self.episode_start_monotonic_ns = time.monotonic_ns()
        if self.scanner:
            self.scan_summary = None
            self.scanner.begin(self.episode_start_monotonic_ns)

        if self.rerun_log:
            self.online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit="300MB")

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        # Author: mingyang - Announce only after the episode was successfully created.
        if self.voice_queue is not None:
            self.voice_queue.put_nowait("开始录制")
        return True  # Return True if the episode is successfully created
        
    def get_episode_elapsed_s(self, sample_monotonic_ns=None):
        if self.episode_start_monotonic_ns is None:
            return None
        if sample_monotonic_ns is None:
            sample_monotonic_ns = time.monotonic_ns()
        return max(
            0.0,
            (sample_monotonic_ns - self.episode_start_monotonic_ns)
            / 1_000_000_000.0,
        )

    def add_item(self, colors, depths=None, states=None, actions=None,
                 tactiles=None, audios=None, sim_state=None,
                 timestamps=None, operator=None, quality=None, unitree=None):
        # Increment the item ID
        self.item_id += 1
        # Create the item data dictionary
        item_data = {
            'idx': self.item_id,
            'colors': colors,
            'depths': depths,
            'states': states,
            'actions': actions,
            'tactiles': tactiles,
            'audios': audios,
            'sim_state': sim_state,
            'timestamps': timestamps,
            'operator': operator,
            'quality': quality,
        }
        # Enqueue the item data
        if self.scanner:
            ns = timestamps["monotonic_ns"] if timestamps else time.monotonic_ns()
            item_data["_unitree"] = self.scanner.frame(ns)
        if unitree is not None:
            if self.scanner:
                raise ValueError("Use either recorder scanner or external scan metadata")
            item_data["_unitree"] = dict(unitree)
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    logger_mp.info(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            # Check if save_episode was triggered
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})

        # Save images
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f'{str(idx).zfill(6)}_{color_key}.jpg'
                if not cv2.imwrite(os.path.join(self.color_dir, color_name), color):
                    logger_mp.info(f"Failed to save color image.")
                item_data['colors'][color_key] = os.path.join('colors', color_name)

        # Save depths
        if depths:
            for idx_depth, (depth_key, depth) in enumerate(depths.items()):
                depth_name = f'{str(idx).zfill(6)}_{depth_key}.jpg'
                if not cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth):
                    logger_mp.info(f"Failed to save depth image.")
                item_data['depths'][depth_key] = os.path.join('depths', depth_name)

        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # Update episode data
        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self.first_item:
                f.write(",\n")
            f.write(json.dumps(item_data, ensure_ascii=False, indent=4))
            self.first_item = False

        # Log data if necessary
        if self.rerun_log:
            curent_record_time = time.time()
            logger_mp.info(f"==> episode_id:{self.episode_id}  item_id:{idx}  current_time:{curent_record_time}")
            self.rerun_logger.log_item_data(item_data)

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        if self.scanner:
            self.scan_summary = self.scanner.finish()
        self.need_save = True  # Set the save flag
        logger_mp.info(f"==> Episode saved start...")

    def discard_episode(self):
        """Discard the active, not-yet-finalized episode and reuse its ID.

        The queue is drained first so the worker cannot still be writing into
        the directory while it is removed. A normally finalized episode is
        never discarded by this method.
        """
        if self.is_available:
            logger_mp.info("==> No active episode to discard.")
            return False
        if self.need_save:
            logger_mp.warning("==> Refusing to discard an episode that is already being finalized.")
            return False

        if self.scanner:
            self.scanner.finish()
        self.item_data_queue.join()

        episode_dir = os.path.abspath(self.episode_dir)
        task_dir = os.path.abspath(self.task_dir)
        episode_name = os.path.basename(episode_dir)
        if (
            os.path.dirname(episode_dir) != task_dir
            or re.fullmatch(r"episode_\d+", episode_name) is None
        ):
            raise RuntimeError(f"Unsafe episode discard path: {episode_dir}")

        if os.path.isdir(episode_dir):
            shutil.rmtree(episode_dir)

        episode_ids = []
        for entry in os.listdir(self.task_dir):
            match = re.fullmatch(r"episode_(\d+)", entry)
            if match and os.path.isdir(os.path.join(self.task_dir, entry)):
                episode_ids.append(int(match.group(1)))
        self.episode_id = max(episode_ids, default=-1)
        self.item_id = -1
        self.episode_start_monotonic_ns = None
        self.is_available = True
        logger_mp.info(f"==> Active episode discarded: {episode_dir}")
        return True

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write("\n]")
            if self.scan_summary is not None:
                f.write(',\n"_unitree": ' + json.dumps(self.scan_summary))
            f.write("\n}")

        # Author: mingyang - Announce after closing data.json, before accepting the next episode.
        if self.voice_queue is not None:
            self.voice_queue.put_nowait("已保存")
        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        logger_mp.info(f"==> Episode saved successfully to {self.json_path}.")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        if not self.is_available and not self.need_save:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()
        if self.scanner:
            self.scanner.close()
        if self.voice_queue is not None:
            self.voice_queue.put_nowait(None)
            self.voice_thread.join(timeout=5.0)
