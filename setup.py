from setuptools import setup

setup(
    name='l4bo',
    version='0.0.1',
    install_requires=[
    ],
    packages=[
    ],
    package_data={
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'energy_reinforce_train=energy.reinforce:train',
        ],
    },
)
