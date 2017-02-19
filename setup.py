
import setuptools


###



classifiers = [
"Development Status :: 4 - Beta",
"Intended Audience :: Science/Research",
"License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
"Programming Language :: C",
"Programming Language :: Python :: 2.7",
"Topic :: Scientific/Engineering :: Bio-Informatics"]



setuptools.setup(
    name         = "pyedr",
    version      = "0.0.1",
    author       = "Justus Schwabedal",
    author_email = "JSchwabedal@gmail.com",
    description  = ("Python library for EKG-derived Respiration."),
    keywords     = "edr",
    url          = "https://github.com/jusjusjus/pyedr",
    packages     = ['pyedr'],
    classifiers	 = classifiers)
