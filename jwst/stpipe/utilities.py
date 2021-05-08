# Copyright (C) 2010 Association of Universities for Research in Astronomy(AURA)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     3. The name of AURA and its representatives may not be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
"""
Utilities
"""
from importlib import import_module
import importlib.util
import inspect
import logging
import os
import re
import sys

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Step classes that are not user-api steps
NON_STEPS = [
    'EngDBLogStep',
    'FunctionWrapper',
    'JwstPipeline',
    'JwstStep',
    'Pipeline',
    'Step',
    'SystemCall',
]


def all_steps():
    """List all classes subclassed from Step

    Returns
    -------
    steps : dict
        Key is the classname, value is the class
    """
    from jwst.stpipe import Step

    jwst = import_module('jwst')
    jwst_fpath = os.path.split(jwst.__file__)[0]

    steps = {}
    for module in load_local_pkg(jwst_fpath):
        more_steps = {
            klass_name: klass
            for klass_name, klass in inspect.getmembers(
                module,
                lambda o: inspect.isclass(o) and issubclass(o, Step)
            )
            if klass_name not in NON_STEPS
        }
        steps.update(more_steps)

    return steps


def load_local_pkg(fpath):
    """Generator producing all modules under fpath

    Parameters
    ----------
    fpath: string
        File path to the package to load.

    Returns
    -------
    generator
        `module` for each module found in the package.
    """
    package_fpath, package = os.path.split(fpath)
    package_fpath_len = len(package_fpath) + 1
    try:
        for module_fpath in folder_traverse(
            fpath, basename_regex=r'[^_].+\.py$', path_exclude_regex='tests'
        ):
            folder_path, fname = os.path.split(module_fpath[package_fpath_len:])
            module_path = folder_path.split('/')
            module_path.append(os.path.splitext(fname)[0])
            module_path = '.'.join(module_path)
            try:
                spec = importlib.util.spec_from_file_location(module_path,module_fpath)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
            except Exception as err:
                logger.debug(f'Cannot load module "{module_path}": {str(err)}')
            else:
                yield module
    except Exception as err:
        logger.debug(f'Cannot complete package loading: Exception occurred: "{str(err)}"')


def folder_traverse(folder_path, basename_regex='.+', path_exclude_regex='^$'):
    """Generator of full file paths for all files
    in a folder.

    Parameters
    ----------
    folder_path: str
        The folder to traverse

    basename_regex: str
        Regular expression that must match
        the `basename` part of the file path.

    path_exclude_regex: str
        Regular expression to exclude a path.

    Returns
    -------
    generator
        A generator, return the next file.
    """
    basename_regex = re.compile(basename_regex)
    path_exclude_regex = re.compile(path_exclude_regex)
    for root, dirs, files in os.walk(folder_path):
        if path_exclude_regex.search(root):
            continue
        for file in files:
            if basename_regex.match(file):
                yield os.path.join(root, file)
