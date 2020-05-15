from setuptools import setup,find_packages
from thexp import __VERSION__
setup(
    name='thexp',
    # 主版本，次版本，修订号？，bug修订号，...待定
    version=__VERSION__,
    description='An useful torch tool for your experiment.',
    url='https://github.com/sailist/thexp',
    author='sailist',
    author_email='sailist@outlook.com',
    license='Apache License 2.0',
    include_package_data = True,
    install_requires = [
      "torch","fire","tensorboard","watchdog","docopt"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='thexp',
    packages=find_packages(),
    entry_points={
        'console_scripts':[
            'thexp = thexp.cli.cli:main'
        ]
      },
)