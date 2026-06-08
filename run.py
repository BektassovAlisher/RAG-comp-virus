import argparse
import subprocess
import sys
import os

def run_api():
    print("🚀 Starting FastAPI backend on http://localhost:8000...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))
    subprocess.run([sys.executable, "app/api/main.py"], env=env)

def run_ui():
    print("🎨 Starting Streamlit UI on http://localhost:8501...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))
    subprocess.run(["streamlit", "run", "app/ui/main.py"], env=env)

def run_evaluation():
    print("📈 Running Ragas evaluation...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))
    
    eval_script = "app/evaluation/evaluate.py"
    if not os.path.exists(eval_script):
        eval_script = "evaluate_rag.py"
    subprocess.run([sys.executable, eval_script], env=env)

def rebuild_db():
    print("🔄 Rebuilding Chroma vector database...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))
    
    cmd = "from app.core.database import get_vectorstore; get_vectorstore(rebuild=True)"
    subprocess.run([sys.executable, "-c", cmd], env=env)

def main():
    parser = argparse.ArgumentParser(description="Launcher for the RAG Security Advisor project.")
    parser.add_init = parser.add_subparsers(dest="command", help="Command to run")
    
    parser.add_init.add_parser("api", help="Start the FastAPI backend server")
    parser.add_init.add_parser("ui", help="Start the Streamlit UI dashboard")
    parser.add_init.add_parser("evaluate", help="Execute Ragas evaluation on the dataset")
    parser.add_init.add_parser("rebuild-db", help="Process PDF and rebuild ChromaDB vector store")

    args = parser.parse_args()

    if args.command == "api":
        run_api()
    elif args.command == "ui":
        run_ui()
    elif args.command == "evaluate":
        run_evaluation()
    elif args.command == "rebuild-db":
        rebuild_db()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()