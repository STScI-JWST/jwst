import os
import pprint

import asdf
from ci_watson.artifactory_helpers import (
    get_bigdata_root,
    get_bigdata,
    BigdataError,
)

from jwst.associations import load_asn


class RegtestData:
    """Defines data paths on Artifactory and data retrieval methods"""

    def __init__(self, env="dev", inputs_root="jwst-pipeline",
        results_root="jwst-pipeline-results", docopy=True,
        input=None, input_remote=None, output=None, truth=None,
        truth_remote=None, remote_results_path=None, test_name=None,
        traceback=None, **kwargs):
        self._env = env
        self._inputs_root = inputs_root
        self._results_root = results_root
        self._bigdata_root = get_bigdata_root()

        self.docopy = docopy

        # Initialize @property attributes
        self.input = input
        self.input_remote = input_remote
        self.output = output
        self.truth = truth
        self.truth_remote = truth_remote

        # No @properties for the following attributes
        self.remote_results_path = remote_results_path
        self.test_name = test_name
        self.traceback = traceback


    def __repr__(self):
        return pprint.pformat(
            dict(input=self.input, output=self.output, truth=self.truth,
            input_remote=self.input_remote, truth_remote=self.truth_remote,
            remote_results_path=self.remote_results_path,
            traceback=self.traceback),
            indent=1
        )

    @property
    def input_remote(self):
        if self._input_remote is not None:
            return os.path.join(*self._input_remote)
        else:
            return None

    @input_remote.setter
    def input_remote(self, value):
        if value:
            self._input_remote = value.split(os.sep)
        else:
            self._input_remote = value

    @property
    def truth_remote(self):
        if self._truth_remote is not None:
            return os.path.join(*self._truth_remote)
        else:
            return None

    @truth_remote.setter
    def truth_remote(self, value):
        if value:
            self._truth_remote = value.split(os.sep)
        else:
            self._truth_remote = value

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        if value:
            self._input = os.path.abspath(value)
        else:
            self._input = value

    @property
    def truth(self):
        return self._truth

    @truth.setter
    def truth(self, value):
        if value:
            self._truth = os.path.abspath(value)
        else:
            self._truth = value

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        if value:
            self._output = os.path.abspath(value)
        else:
            self._output = value

    @property
    def bigdata_root(self):
        return self._bigdata_root

    # The methods
    def get_data(self, path=None, docopy=None):
        """Copy data from Artifactory remote resource to the CWD

        Updates self.input and self.input_remote upon completion
        """
        if path is None:
            path = self.input_remote
        else:
            self.input_remote = path
        if docopy is None:
            docopy = self.docopy
        self.input = get_bigdata(self._inputs_root, self._env, path,
            docopy=docopy)

        return self.input

    def get_truth(self, path=None, docopy=None):
        """Copy truth data from Artifactory remote resource to the CWD/truth

        Updates self.truth and self.truth_remote on completion
        """
        if path is None:
            path = self.truth_remote
        else:
            self.truth_remote = path
        if docopy is None:
            docopy = self.docopy
        os.makedirs('truth', exist_ok=True)
        os.chdir('truth')
        try:
            self.truth = get_bigdata(self._inputs_root, self._env, path,
                docopy=docopy)
            self.truth_remote = os.path.join(self._inputs_root, self._env, path)
        except BigdataError:
            os.chdir('..')
            raise
        os.chdir('..')

        return self.truth

    def get_asn(self, path=None, docopy=None):
        """Copy association and association members from Artifactory remote
        resource to the CWD/truth.

        Updates self.input and self.input_remote upon completion
        """
        if path is None:
            path = self.input_remote
        else:
            self.input_remote = path
        if docopy is None:
            docopy = self.docopy

        # Get the association JSON file
        self.input = get_bigdata(self._inputs_root, self._env, path,
            docopy=docopy)

        # Get each member in the association as well
        with open(self.input) as fp:
            asn = load_asn(fp)
        for product in asn['products']:
            for member in product['members']:
                fullpath = os.path.join(
                    os.path.dirname(self.input_remote),
                    member['expname'])
                get_bigdata(self._inputs_root, self._env, fullpath,
                    docopy=self.docopy)

    def to_asdf(self, path):
        tree = eval(str(self))
        af = asdf.AsdfFile(tree=tree)
        af.write_to(path)

    @classmethod
    def open(cls, filename):
        with asdf.open(filename) as af:
            return cls(**af.tree)
