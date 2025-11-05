import subprocess
import sys
import os
from pathlib import Path

def check_tesseract():
    """Check if Tesseract is installed"""
    try:
        subprocess.run(['tesseract', '--version'],
                      capture_output=True, check=True)
        print("✅ Tesseract is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Tesseract is not installed or not in PATH")
        print("📝 Installation instructions:")
        print("   - macOS: brew install tesseract tesseract-lang")
        print("   - Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng")
        print("   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def check_python_packages():
    """Vérifier et installer les packages Python requis"""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print("❌ requirements.txt file not found")
        return False

    try:
        print("📦 Checking Python dependencies...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
        ], check=True)
        print("✅ All dependencies are installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_env_file():
    """Check for the .env file in the project root"""
    env_file = Path(__file__).parent.parent / ".env"

    if not env_file.exists():
        print("⚠️  .env file not found in project root")
        print("💡 Create a .env file with your API key:")
        print("   API_KEY_OPENAI=your_api_key_here")
        return False

    print(f"✅ .env file found at {env_file}")
    return True

def launch_streamlit():
    """Lancer l'application Streamlit"""
    app_file = Path(__file__).parent / "streamlit_app.py"

    if not app_file.exists():
        print(f"❌ Fichier {app_file} non trouvé")
        return False

    try:
        print("🚀 Launching Streamlit application...")
        print("🌐 The app will be available at http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop")

        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', str(app_file),
            '--server.address', 'localhost',
            '--server.port', '8501',
            '--browser.gatherUsageStats', 'false'
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error while launching: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True

def main():
    """Fonction principale"""
    print("🔧 Preparing the invoice analysis application")
    print("=" * 50)

    # Vérifications préliminaires
    checks_passed = True

    # 1. Tesseract
    if not check_tesseract():
        checks_passed = False

    #if not check_python_packages():
        #checks_passed = False

    # 3. Fichier .env
    env_ok = check_env_file()

    if not checks_passed:
        print("\n❌ Missing prerequisites. Please fix the errors above.")
        return

    if not env_ok:
        print("\n⚠️  Incomplete configuration. Set your API key in the .env file")
        print("   You can also enter it directly in the interface.")

    print("\n" + "=" * 50)

    # Lancement de l'application
    launch_streamlit()

if __name__ == "__main__":
    main()