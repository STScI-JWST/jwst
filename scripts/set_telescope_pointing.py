#!/usr/bin/env python
"""
Set the initial world coordinate system for JWST exposures

The JWST engineering database is queried for the JWST observatory
orientation parameters, and converts that orientation to a WCS
for a list of exposures.

Copyright
---------
Copyright (C) 2010-2011 Association of Universities for Research in Astronomy (AURA)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    3. The name of AURA and its representatives may not be used to
      endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

import argparse
import logging

import jwst.lib.set_telescope_pointing as stp

# Configure logging
logger = logging.getLogger('jwst')
logger.propagate = False
logger_handler = logging.StreamHandler()
logger.addHandler(logger_handler)
logger_format_debug = logging.Formatter('%(levelname)s:%(filename)s::%(funcName)s: %(message)s')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Update basic WCS information in JWST exposures from the engineering database.'
    )
    parser.add_argument(
        'exposure', type=str, nargs='+',
        help='List of JWST exposures to update.'
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Increase verbosity. Specifying multiple times adds more output.'
    )
    parser.add_argument(
        '--method',
        type=stp.Methods, choices=list(stp.Methods), default=stp.Methods.default,
        help='Algorithmic method to use. Default: %(default)s'
    )
    parser.add_argument(
        '--tolerance', type=int, default=60,
        help='Seconds beyond the observation time to search for telemetry. Default: %(default)s'
    )
    parser.add_argument(
        '--allow-default', action='store_true',
        help='If pointing information cannot be determine, use header information.'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Perform all actions but do not save the results'
    )
    parser.add_argument(
        '--siaf', type=str, default=None,
        help='SIAF PRD sqlite database file. If not specified, default is to use `$XML_DATA/prd.db`'
    )
    parser.add_argument(
        '--engdb_url', type=str, default=None,
        help=('URL of the engineering database.'
              ' If not specified, the environmental variable "ENG_BASE_URL" is used.'
              ' Otherwise, a hardwired default is used.')
    )
    parser.add_argument(
        '--transpose_j2fgs', action='store_false',
        help='Transpose the J2FGS matrix'
    )

    args = parser.parse_args()

    # Set output detail.
    level = stp.LOGLEVELS[min(len(stp.LOGLEVELS) - 1, args.verbose)]
    logger.setLevel(level)
    if level <= logging.DEBUG:
        logger_handler.setFormatter(logger_format_debug)

    # Calculate WCS for all inputs.
    for filename in args.exposure:
        logger.info(
            '\n------'
            'Setting pointing for {}'.format(filename)
        )
        try:
            stp.add_wcs(
                filename,
                siaf_path=args.siaf,
                engdb_url=args.engdb_url,
                tolerance=args.tolerance,
                allow_default=args.allow_default,
                dry_run=args.dry_run,
                method=args.method,
                j2fgs_transpose=args.transpose_j2fgs
            )
        except ValueError as exception:
            logger.info('Cannot determine pointing information: ' + str(exception))
