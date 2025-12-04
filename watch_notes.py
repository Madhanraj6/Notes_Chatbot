import os
import sys
import subprocess
import threading
import time
from watchdog.events import FileSystemEventHandler

# Config
VAULT_PATH = "./Obsidian"
INGEST_SCRIPT = "ingest.py"
WATCH_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".pptx", ".jpg", ".jpeg", ".png"}
DEBOUNCE_SECONDS = 3.0

class NotesChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.timer = None
        self.lock = threading.Lock()
        self.pending_changes = set()

    def on_any_event(self, event):
        if event.is_directory:
            return

        ext = os.path.splitext(event.src_path)[1].lower()
        if ext not in WATCH_EXTENSIONS:
            return

        print(f"[CHANGE] {event.event_type} — {event.src_path}")

        with self.lock:
            self.pending_changes.add(event.src_path)
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(DEBOUNCE_SECONDS, self.run_ingest)
            self.timer.start()

    def run_ingest(self):
        with self.lock:
            if not self.pending_changes:
                return
            changed_count = len(self.pending_changes)
            self.pending_changes.clear()

        print(f"[INFO] Re-ingesting {changed_count} changed file(s)...")

        try:
            result = subprocess.run(
                [sys.executable, INGEST_SCRIPT],
                check=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.stdout.strip():
                print(f"[OUTPUT] {result.stdout.strip()}")
            print("[SUCCESS] Ingestion complete.")

        except subprocess.TimeoutExpired:
            print("[ERROR] Ingestion timed out after 5 minutes.")
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] Ingestion script failed (exit {e.returncode}).")
            if e.stdout.strip():
                print(f"[STDOUT] {e.stdout.strip()}")
            if e.stderr.strip():
                print(f"[STDERR] {e.stderr.strip()}")
        except Exception as e:
            print(f"[FAIL] Unexpected error during ingestion: {e}")
        finally:
            print("-" * 50)

def main():
    if not os.path.exists(VAULT_PATH):
        print(f"[ERROR] Vault path '{VAULT_PATH}' does not exist.")
        print("Please ensure the Obsidian folder is mounted correctly.")
        return 1

    print(f"[INFO] Watching folder: {os.path.abspath(VAULT_PATH)}")
    print(f"[INFO] File types: {', '.join(WATCH_EXTENSIONS)}")
    print(f"[INFO] Debounce: {DEBOUNCE_SECONDS} seconds")
    print("=" * 50)

    try:
        from watchdog.observers.polling import PollingObserver as ObserverClass
        print("[INFO] Using PollingObserver for Docker compatibility.")
    except ImportError:
        from watchdog.observers import Observer as ObserverClass
        print("[INFO] Using default Watchdog observer.")

    observer = ObserverClass()
    handler = NotesChangeHandler()

    try:
        observer.schedule(handler, path=VAULT_PATH, recursive=True)
        observer.start()
        print("[START] File watcher running — Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] Stopping watcher...")
        observer.stop()
    except Exception as e:
        print(f"[ERROR] Watcher crashed: {e}")
        observer.stop()
    finally:
        observer.join()
        print("[DONE] Watcher stopped.")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code or 0)
