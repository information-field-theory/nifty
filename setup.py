## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2013 Max-Planck-Society
##
## Author: Marco Selig
## Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.

from distutils.core import setup
import os

setup(name="nifty",
      version="0.6.0",
      description="Numerical Information Field Theory",
      author="Marco Selig",
      author_email="mselig@mpa-garching.mpg.de",
      url="http://www.mpa-garching.mpg.de/ift/nifty/",
      packages=["nifty"],
      package_dir={"nifty": ""},
      package_data={"nifty": ["demos/demo_excaliwir.py",
                              "demos/demo_faraday.py",
                              "demos/demo_faraday_map.npy",
                              "demos/demo_wf1.py",
                              "demos/demo_wf2.py"]},
      data_files=[(os.path.expanduser('~') + "/.nifty", ["nifty_config"])])