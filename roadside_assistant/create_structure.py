from pathlib import Path

def create_folder_structure(base_path="."):
    """Creates the roadside_assistant folder structure."""
    base = Path(base_path)
    
    # Define the structure
    structure = {
        "agents": [
            "__init__.py",
            "base.py",
            "intake_agent.py",
            "rag_agent.py",
            "dispatch_agent.py",
            "monitoring_agent.py",
            "extraction_agent.py"
        ],
        "services": [
            "rag_service.py",
            "dispatch_service.py",
            "monitoring_service.py"
        ],
        "tools": [
            "information_gathering.py",
            "knowledge_search.py",
            "dispatch_tools.py",
            "data_extraction.py"
        ],
        "main.py": None  # This will be a file
    }

    # Create directories and files
    for folder, files in structure.items():
        if files is not None:  # It's a directory
            dir_path = base / folder
            dir_path.mkdir(exist_ok=True)
            print(f"Created directory: {dir_path}")
            
            for file in files:
                file_path = dir_path / file
                file_path.touch()
                print(f"Created file: {file_path}")
        else:  # It's a file in the root
            file_path = base / folder
            file_path.touch()
            print(f"Created file: {file_path}")

    print("\nFolder structure created successfully!")

if __name__ == "__main__":
    create_folder_structure()
    