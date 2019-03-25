import sys
from setuptools import setup


extras_require = dict(
    shdp=['matplotlib']
)

setup(
    name='deep-trends',
    version='0.0.0a',
    description='Simple framework for trend analysis',
    url='https://github.com/EemeliSaari/DeepTrends/',
    maintainer='Eemeli Saari',
    maintainer_email='saari.eemeli@gmail.com',
    license='GNU2',
    install_requires=open('requirements.txt').read().strip().split('\n'),
    extras_require=extras_require,
    packages=['',
              '',
              '',
              ''],
    long_description=open('README.md').read(),
    entry_ponts='''
        [console_scripts]
        parsepdf=deep_trends.cli.parse_pdfs:go
        docs2topic=deep_trends.cli.generate_topics:go
    ''',
    zip_safe=False
)