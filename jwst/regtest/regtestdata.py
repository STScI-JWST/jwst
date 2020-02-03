import os
import pprint
import shutil

import asdf
from astropy.io.fits.diff import FITSDiff
from ci_watson.artifactory_helpers import (
    get_bigdata_root,
    get_bigdata,
    BigdataError,
)

from jwst.associations import AssociationNotValidError, load_asn
from jwst.lib.suffix import replace_suffix
from jwst.pipeline.collect_pipeline_cfgs import collect_pipeline_cfgs
from jwst.stpipe import Step


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

        # Initialize non-initialized attributes
        self.asn = None

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
        self.input_remote = os.path.join(self._inputs_root, self._env, path)

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

    def get_asn(self, path=None, docopy=True, get_members=True):
        """Copy association and association members from Artifactory remote
        resource to the CWD/truth.

        Updates self.input and self.input_remote upon completion

        Parameters
        ----------
        path: str
            The remote path

        docopy : bool
            Switch to control whether or not to copy a file
            into the test output directory when running the test.
            If you wish to open the file directly from remote
            location or just to set path to source, set this to `False`.
            Default: `True`

        get_members: bool
            If an association is the input, retrieve the members.
            Otherwise, do not.
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
        with open(self.input) as fp:
            asn = load_asn(fp)
            self.asn = asn

        # Get each member in the association as well
        if get_members:
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


def run_step_from_dict(rtdata, **step_params):
    """Run Steps with given parameter

    Parameters
    ----------
    rtdata: RegtestData
        The artifactory instance

    step_params: dict
        The parameters defining what step to run with what input

    Returns
    -------
    rtdata: RegtestData
        Updated `RegtestData` object with inputs set.

    Notes
    -----
    `step_params` looks like this:
    {
        'input_path': str or None  # The input file path, relative to artifactory
        'step': str                # The step to run, either a class or a config file
        'args': list,              # The arguments passed to `Step.from_cmdline`
    }
    """

    # Get the data. If `step_params['input_path]` is not
    # specified, the presumption is that `rtdata.input` has
    # already been retrieved.
    input_path = step_params.get('input_path', None)
    if input_path:
        try:
            rtdata.get_asn(input_path)
        except AssociationNotValidError:
            rtdata.get_data(input_path)

    # Figure out whether we have a config or class
    step = step_params['step']
    if step.endswith(('.asdf', '.cfg')):
        step = os.path.join('config', step)

    # Run the step
    collect_pipeline_cfgs('config')
    full_args = [step, rtdata.input]
    full_args.extend(step_params['args'])

    Step.from_cmdline(full_args)

    return rtdata


def run_step_from_dict_mock(rtdata, source,  **step_params):
    """Pretend to run Steps with given parameter but just copy data

    For long running steps where the result already exists, just
    copy the data from source

    Parameters
    ----------
    rtdata: RegtestData
        The artifactory instance

    step_params: dict
        The parameters defining what step to run with what input

    source: Path-like folder
        The folder to copy from. All regular files are copied.

    Returns
    -------
    rtdata: RegtestData
        Updated `RegtestData` object with inputs set.

    Notes
    -----
    `step_params` looks like this:
    {
        'input_path': str or None  # The input file path, relative to artifactory
        'step': str                # The step to run, either a class or a config file
        'args': list,              # The arguments passed to `Step.from_cmdline`
    }
    """

    # Get the data. If `step_params['input_path]` is not
    # specified, the presumption is that `rtdata.input` has
    # already been retrieved.
    input_path = step_params.get('input_path', None)
    if input_path:
        try:
            rtdata.get_asn(input_path)
        except AssociationNotValidError:
            rtdata.get_data(input_path)

    # Copy the data
    for file_name in os.listdir(source):
        file_path = os.path.join(source, file_name)
        if os.path.isfile(file_path):
            shutil.copy(file_path, '.')

    return rtdata


def is_like_truth(rtdata, fitsdiff_default_kwargs, output, truth_path, is_suffix=True):
    """Compare step outputs with truth

    Parameters
    ----------
    rtdata: RegtestData
        The artifactory object from the step run.

    fitsdiff_default_kwargs: dict
        The `fitsdiff` keyword arguments

    output: str
        The suffix or full file name to check on.

    truth_path: str
        Location of the truth files.

    is_suffix: bool
        Interpret `output` as just a suffix on the expected output root.
        Otherwise, assume it is a full file name
    """

    # If given only a suffix, get the root to change the suffix of.
    # If the input was an association, the output should be the name of the product
    # Otherwise, output is based on input.
    if is_suffix:
        suffix = output
        if rtdata.asn:
            output = rtdata.asn['products'][0]['name']
        else:
            output = os.path.splitext(os.path.basename(rtdata.input))[0]
        output = replace_suffix(output, suffix) + '.fits'
    rtdata.output = output

    rtdata.get_truth(os.path.join(truth_path, output))

    diff = FITSDiff(rtdata.output, rtdata.truth, **fitsdiff_default_kwargs)
    assert diff.identical, diff.report()
