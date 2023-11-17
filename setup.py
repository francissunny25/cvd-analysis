from setuptools import find_packages, setup
from typing import List


##TOD remove hyphen and make it personalised
HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
    Retrieves the list of required libraries
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name = 'cvd-analysis',
    version='0.0.1',
    author='Francis Sunny',
    author_email='francis.sunny.25@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
