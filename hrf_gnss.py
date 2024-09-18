#!/usr/bin/env python
"""
Record GNSS data from a particular band/code
using HackRF SDR, and output in SigMF format
"""
from subprocess import Popen, PIPE, STDOUT
import re
import argparse
import os
import json
from datetime import datetime, timezone

import numpy as np
import sigmf
from sigmf import SigMFFile

def freq_ctr_and_bw(bandcode):
    """
    Convert a GNSS band code to the corresponding center frequency and bandwidths (all in float MHz),
    for example:
    GPS L1 Band: 1575.42 MHz with a bandwidth of 15.345 MHz
    GPS L2 Band: 1227.6 MHz with a bandwidth of 11 MHz
    GPS L5 Band: 1176.45 MHz with a bandwidth of 12.5 MHz
    Note that these bandwidths are the full allocated bandwidth: receiving a narrower bandwidth may be sufficient.
    Refer to ESA GPS band plan notes: https://gssc.esa.int/navipedia/index.php/GPS_Signal_Plan

    :param bandcode: A short string code for the band, eg 'L5' or 'L1CA'
    :return: (center_freq_mhz, full_bandwidth_mhz, sampling_bandwidth_mhz, baseband_filter_mhz)
    """
    match bandcode:
        case 'L2' | 'L2C' | 'L2CM':
            # The exact L2C chip rate is 1.023 Msps (2x 511.5 Ksps multiplexed)
            # "The CM code is 10,230 chips long, repeating every 20 ms.
            #  The CL code is 767,250 chips long, repeating every 1,500 ms.
            #  Each signal is transmitted at 511,500 chips per second (chip/s);
            # however, they are multiplexed together to form a 1,023,000-chip/s signal."
            return 1227.6000, 11.000, 4.0, 2.5
        case 'L3':
            return 1381.0500, 15.345, 2.0, 2.0
        case 'L4':
            return 1379.9133, 15.345, 2.0, 2.0
        case 'L5':
            # L5 has 10.23 MHz chipping rate (10.23E6 chips /sec)
            # L5 has a code length of 10.23E3 chips
            return 1176.4500, 12.500, 12.50, 12.0
        case 'L5I':
            return 1176.4500, 20.4600, 20.0, 20.0
        case 'B2a':
            return 1176.4500, 12.500, 2.0, 2.0
        case 'E5a':
            return 1176.4500, 20.460, 2.0, 2.0
        case 'E5b' | 'B2b':
            return 1207.1400, 20.460, 2.0, 2.0
        case 'E6':
            return 1278.7500, 40.920, 2.0, 2.0
        case 'B3':
            return 1286.5300, 10.000, 2.0, 2.0
        case 'E1':
            return 1575.4200, 24.552, 2.0, 2.0
        case 'H1':
            # Useful for recording hydrogen line in the same manner as we record GNSS
            return 1420.4000, 12.000, 2.0, 2.0
        case 'L1':
            # adequate for most L1 GPS uses
            return 1575.4200, 15.345, 2.0, 2.0
        case 'L1CA':
            # exact chip rate for C/A is 1.023E6 chips/sec
            # number of chips is 1023
            # seconds/chip is 1E-6, sampled at Nyquist: 2 MHz
            return 1575.4200, 15.345, 4.0, 2.0
        case 'L1PY':
            # chip rate for P(Y) code is 10.23E6 chips/sec?
            # number of chips is 10230 ?
            return 1575.4200, 15.345, 10.23, 10.0
        case 'L1M':
            # Exact chip rate for M code is 5.115 MHz, and sub-carrier is 10.23 MHz?
            return 1575.4200, 15.345, 10.23, 5.115
        case 'V1':
            # simulate receving a "virtual" white noise signal on the L1 band
            return 1575.4200, 24.0, 10.0, 10.0
        case 'RCM':
            # receive from the Radarsat Constellation (RCM) or similar
            return 5405.0000, 25.0, 25.0, 25.0
        case 'KALX' | _:
            return 90.7000, 5.000, 2.0, 0.2 # KALX.berkeley.edu

def band_gains(bandcode):
    """
    Adjust the LNA (IF) and baseband gains according to the band
    Great Scott mentions that there are three HackRF adjustments for gain:
    - RF RX amp (either on or off, up to 14 dB of gain)
    - IF ("LNA") gain:  0-40dB, 8dB steps
    - Baseband ("VGA") gain: 0-62dB, 2dB steps
    We don't use the RF amp, instead using the fine IF + BB controls

    :param bandcode:
    :return: IF (LNA) gain, baseband (VGA) gain
    """
    match bandcode:
        case 'L1'|'L1CA':
            return 40, 24
        case 'L1M':
            return 40, 20
        case 'L2' | 'L2CM' | 'L2C' :
            return 40, 16
        case 'L5' | 'L5I' :
            return 32, 24
        case 'KALX':
            return 40, 24
        case _:
            return 40, 40

def main():
    parser = argparse.ArgumentParser(description='Grab some GNSS data using hackrf_transfer')
    parser.add_argument('--band', '-b', dest='bandcode', default='L1',
                        choices=['L1', 'L2', 'L3', 'L4', 'L5',
                                 'L1CA','L1M', 'L2C', 'L2CM', 'L5I',
                                 'B2a', 'B2c', 'B3',
                                 'E1', 'E5a', 'E5b', 'E6',
                                 'H1','KALX', 'V1', 'RCM',
                                 ],
                        help='Short code string for the GNSS band selected (default: L1)')
    parser.add_argument('--duration', '-d',  type=int, default=30,
                        help='Duration to capture, in seconds (default: 30)')
    parser.add_argument('--serial_num', '-sn',   default=None,
                        help='Specific HackRF serial number to use (default: None)')

    args = parser.parse_args()
    bandcode = args.bandcode
    duration_seconds = args.duration
    specific_hrf_sn = args.serial_num

    print(f'band {bandcode} duration {duration_seconds}')

    freq_ctr_mhz, full_bandwidth, sampling_bw_mhz, true_bb_bw_mhz = freq_ctr_and_bw(bandcode)
    print(f"ctr: {freq_ctr_mhz} bw: {sampling_bw_mhz}")

    if sampling_bw_mhz < 2.0:
        sampling_bw_mhz = 2.0
    sample_rate_hz = int(sampling_bw_mhz * 1E6)
    ctr_freq_hz = int(freq_ctr_mhz * 1E6)
    # 1.75/2.5/3.5/5/5.5/6/7/8/9/10/12/14/15/20/24/28MHz, default <= 0.75 * sample_rate_hz.
    bb_filter_mhz = true_bb_bw_mhz
    if bb_filter_mhz < 1.75:
        bb_filter_mhz = 1.75
    baseband_filter_bw_hz = int(bb_filter_mhz*1E6)

    if_lna_gain_db, baseband_gain_db = band_gains(bandcode)

    n_samples = int(duration_seconds * sample_rate_hz)

    true_bb_bw_hz = int(true_bb_bw_mhz*1E6)
    print(f"true_bb_bw_hz: {true_bb_bw_hz}")
    half_baseband_bandwidth = int(true_bb_bw_hz / 2)
    freq_lower_edge = int(ctr_freq_hz - half_baseband_bandwidth)
    freq_upper_edge = int(ctr_freq_hz + half_baseband_bandwidth)
    # figure out where to put the output files automatically
    file_number = 1
    path_stem = f'hrf_gnss_{bandcode}_{duration_seconds}s'
    data_out_path = f'{path_stem}_{file_number:04d}.sigmf-data'
    while os.path.isfile(data_out_path):
        file_number += 1
        data_out_path = f'{path_stem}_{file_number:04d}.sigmf-data'
    meta_out_path = f'{path_stem}_{file_number:04d}.sigmf-meta'

    # sample SN: 0000000000000000c66c63dc2d898983
    opt_str = f"-f {ctr_freq_hz} -a 1 -l {if_lna_gain_db} -g {baseband_gain_db} -b {baseband_filter_bw_hz} -s {sample_rate_hz} -n {n_samples}  -B -r {data_out_path}"
    if specific_hrf_sn is None:
        cmd_str = f"hackrf_transfer {opt_str}"
    else:
        cmd_str = f"hackrf_transfer -d {specific_hrf_sn} {opt_str}"

    print(f"START:\n{cmd_str} ")

    # Regex to match and extract numeric values
    regex = r"[-+]?\d*\.\d+|\d+"

    total_power = float(0)
    step_count = 0
    line_count = 0
    capture_start_utc = None
    if bandcode != 'V1':
        with (Popen([cmd_str], stdout=PIPE, stderr=STDOUT, text=True, shell=True) as proc):
            for line in proc.stdout:
                if line_count > 6: # skip command startup lines
                    if capture_start_utc is None:
                        capture_start_utc = datetime.utcnow().isoformat()+'Z'

                    numeric_values = re.findall(regex, line)
                    if numeric_values is not None and len(numeric_values) == 7:
                        # 8.1 MiB / 1.000 sec =  8.1 MiB/second, average power -2.0 dBfs, 14272 bytes free in buffer, 0 overruns, longest 0 bytes
                        # ['8.1', '1.000', '8.1', '-2.0', '14272', '0', '0']
                        print(numeric_values)
                        step_power = numeric_values[3]
                        total_power += float(step_power)
                        step_count += 1
                    else:
                        # read all the stdout until finished, else data out files are not flushed
                        continue
                line_count += 1
    else:
        # save a fake signal file
        total_interleaved_values = n_samples * 2
        # I and Q values should be completely independent
        one_big_chunk = np.random.random(total_interleaved_values)*np.iinfo(np.int8).max
        one_big_chunk.astype('int8').tofile(data_out_path)
        total_power = -1.0
        step_count = 1

    rc = proc.returncode
    if 0 != rc:
        print(f"hackrf_transfer failed with result code: {rc}")

    avg_power = total_power / float(step_count)
    print(f"avg_power: {avg_power:02.3f} (dBFS)")

    # TODO look at using the SigMFFile object, directly, instead
    meta_info_dict = {
    "global": {
        SigMFFile.DATATYPE_KEY: 'ci8',
        SigMFFile.SAMPLE_RATE_KEY: int(f'{sample_rate_hz}'),
        SigMFFile.HW_KEY: "HackRF, HT004a boost amp, bias tee, active ceramic patch antenna",
        SigMFFile.AUTHOR_KEY: 'Todd Stellanova',
        SigMFFile.VERSION_KEY: f'{sigmf.__version__}', 
        SigMFFile.DESCRIPTION_KEY: f'GNSS {bandcode} recorded using hackrf_transfer',
        SigMFFile.RECORDER_KEY: 'hackrf_transfer',
        'antenna:type': 'patch',
        'stellanovat:sdr': 'HackRF',
        'stellanovat:sdr_sn': f'{specific_hrf_sn}',
        'stellanovat:LNA': 'active_antenna',
        'stellanovat:LNA_pwr': 'bias_tee',
        'stellanovat:boost_amp': 'HT004a',
        'stellanovat:boost_amp_pwr': 'USB-C',
    },
    "captures": [
        {
            SigMFFile.START_INDEX_KEY: 0,
            SigMFFile.FREQUENCY_KEY: int(f'{ctr_freq_hz}'), 
            SigMFFile.DATETIME_KEY: f'{capture_start_utc}',
            'stellanovat:if_gain_db': int(f'{if_lna_gain_db}'),
            'stellanovat:bb_gain_db': int(f'{baseband_gain_db}'),
            'stellanovat:sdr_rx_amp_enabled': 0,
            "stellanovat:recorder_command": f'{cmd_str}',
        }
    ],
    "annotations": [
        {
            SigMFFile.START_INDEX_KEY: 0,
            SigMFFile.LENGTH_INDEX_KEY: int(f'{n_samples}'),
            SigMFFile.FHI_KEY: int(f'{freq_upper_edge}'),
            SigMFFile.FLO_KEY: int(f'{freq_lower_edge}'),
            SigMFFile.LABEL_KEY: f'GNSS {bandcode}',
            'stellanovat:avg_dbfs':float(f'{avg_power:0.3f}'),
        }
    ]
    }

    meta_json = json.dumps(meta_info_dict, indent=2)
    # print(meta_json)
    
    with open(meta_out_path, "w") as meta_outfile:
        meta_outfile.write(meta_json)

    print(f"wrote {meta_out_path}")



if __name__ == "__main__":
    main()
