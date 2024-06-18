import os
import subprocess


def generate_requirements():
    # Ensure we're in the correct directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    # Create or update requirements.txt
    with open('requirements.txt', 'w') as f:
        subprocess.call(['pip', 'freeze'], stdout=f)

    print("requirements.txt generated successfully!")


if __name__ == "__main__":
    generate_requirements()
