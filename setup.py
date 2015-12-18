try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'A few image upscaling algorithms for use with scikit-image',
    'author': 'Philip Blair',
    'author_email': 'peblairman@gmail.com',
    'version': '0.9',
    'install_requires': ['scikit-image', 'numpy', 'matplotlib'],
    'packages': ['skupscale'],
    'name': 'skupscale',
    'download_url': 'https://github.com/belph/skupscale'
}

setup(**config)
