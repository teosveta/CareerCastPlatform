#!/usr/bin/env python3
"""
CareerCast Platform Server Launcher
Run this script to start the CareerCast Platform with proper setup.
"""

import os
import sys
from pathlib import Path
import uvicorn


def setup_environment():
    """Setup the environment for running the CareerCast Platform."""

    # Get the project root (where this script is located)
    script_dir = Path(__file__).parent
    project_root = script_dir

    print("🚀 CareerCast Platform Setup")
    print("=" * 50)
    print(f"📁 Project root: {project_root}")

    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"✅ Added {project_root} to Python path")

    # Create necessary directories
    directories_to_create = [
        project_root / "dashboard" / "templates",
        project_root / "dashboard" / "static" / "css",
        project_root / "dashboard" / "static" / "js",
        project_root / "logs",
        project_root / "data" / "processed",
    ]

    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Ensured directory exists: {directory}")

    # Check for required files
    required_files = {
        "HTML Template": project_root / "dashboard" / "templates" / "index.html",
        "CSS Styles": project_root / "dashboard" / "static" / "css" / "styles.css",
        "JavaScript": project_root / "dashboard" / "static" / "js" / "main.js",
        "Main API": project_root / "src" / "api" / "main.py"
    }

    print("\n📋 File Check:")
    all_files_exist = True
    for name, file_path in required_files.items():
        if file_path.exists():
            print(f"✅ {name}: {file_path}")
        else:
            print(f"❌ {name}: MISSING - {file_path}")
            all_files_exist = False

    if not all_files_exist:
        print("\n⚠️  Some required files are missing!")
        print("Please ensure you have:")
        print("1. Saved the HTML template to dashboard/templates/index.html")
        print("2. Saved the CSS file to dashboard/static/css/styles.css")
        print("3. Saved the JavaScript file to dashboard/static/js/main.js")
        print("\nYou can still run the server, but the frontend may not work properly.")

        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            return False

    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking Dependencies:")

    required_packages = [
        "fastapi",
        "uvicorn",
        "jinja2",
        "python-multipart"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def main():
    """Main function to start the server."""
    print("🎯 CareerCast Platform - Skills Demand Forecasting")
    print("=" * 60)

    # Setup environment
    if not setup_environment():
        print("❌ Setup failed. Exiting.")
        return

    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies. Please install them first.")
        return

    print("\n🌐 Starting CareerCast Platform Server...")
    print("=" * 50)

    # Import and run the FastAPI app
    try:
        # Change to project root directory
        os.chdir(Path(__file__).parent)

        print("📍 Access Points:")
        print("   🌐 Frontend:     http://localhost:8000/")
        print("   📚 API Docs:     http://localhost:8000/api/docs")
        print("   💚 Health Check: http://localhost:8000/api/health")
        print("   📊 Dashboard:    http://localhost:8000/dashboard")
        print("\n🔥 Server starting...")
        print("   Press Ctrl+C to stop the server")
        print("=" * 50)

        # Run the FastAPI application
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["src", "dashboard"],
            workers=1
        )

    except ImportError as e:
        print(f"❌ Failed to import the FastAPI app: {e}")
        print("\nTrying alternative import method...")

        try:
            # Alternative method - import directly
            from src.api.main import app
            uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
        except Exception as e2:
            print(f"❌ Alternative import also failed: {e2}")
            print("\nPlease check that your main.py file is correctly set up.")

    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user. Goodbye!")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()