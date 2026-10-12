"""Entry point for the bounded, checkpointed draft rebuild."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capstone_person1.streaming_rebuild import main

if __name__ == "__main__":
    main()
