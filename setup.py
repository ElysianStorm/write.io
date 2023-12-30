import setuptools

# This file allows to setup the description of the project on open-source, if committed to pipy etc.
# Reads the information present in README.md and adds additional metadata along with the project description for pbulication related information.
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

# Git Repo name
REPO_NAME = "write.io"
# Git Author/Username
AUTHOR_USER_NAME = "ElysianStorm"
# Local project-repo src name ('project_name' in src\{project_name})
SRC_REPO = "write_io" 
# Git Linked Email Address
AUTHOR_EMAIL = "alok212@hotmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    description="Deep Learning Model based on RNN and CRNN to convert handwritten texts to computer readable text.",
    Long_description=long_description,
    long_description_content="text/markdown",
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR_USER_NAME,
    maintainer_email=AUTHOR_EMAIL,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    }
)
