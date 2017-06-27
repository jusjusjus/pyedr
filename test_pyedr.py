#!/usr/bin/python3

import pyedr
from pyedr.ekg import Ekg
import unittest

class TestPyedr(unittest.TestCase):

     def test_synthetic(self):
         fs = 250
         num_samples = 128
         num_of_segments = 1
         f_HR = 1.2
         q = 0.0
         ds = pyedr.Dataset(subject_ids=['synthetic'], sampling_rate=fs,
                            num_of_segments=num_of_segments, heart_rate=f_HR, hr_stdev_factor=q,
                            rsa_strength=0.0, esk_strength=0.0,
                            EKG_noise_strength=0.0, EKG_fluctuation_strength=0.0,
                            num_samples=num_samples)
         output = ds.get_data(normalize=None)
         self.assertEqual(type(output), list)
         self.assertEqual(len(output), num_of_segments)
         self.assertEqual(len(output[0]), 2)
         self.assertEqual(len(output[0][0]), num_samples)


     def test_ekg(self):
         expected_result = 4853
         seed = 42
         num_seg = 2
         fs = 250
         ds = pyedr.Dataset(subject_ids=['synthetic'], num_of_segments=num_seg,
                            hr_stdev_factor=0.0, EKG_fluctuation_strength=0.0, esk_strength=0.0, rsa_strength=0.0,
                            seed=seed, sampling_rate=fs)
         data = ds.get_data(normalize=None)
         ecg = [d[0] for d in data]
         ekg = Ekg(ecg, sampling_rate=fs)
         ekg.get_all_R_peaks()
         self.assertEqual(ekg.R_peaks[0][-1], expected_result)
         self.assertEqual(len(ekg.R_peaks), 2)
         self.assertEqual(len(ekg.R_peaks[0]),  20)

if __name__ == "__main__":
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.makeSuite(TestPyedr))
    # test_suite.addTest(doctest.DocTestSuite(...))
    unittest.TextTestRunner().run(test_suite)
