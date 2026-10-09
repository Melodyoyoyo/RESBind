import unittest
import numpy as np
from resbind_server.pockets import candidate_pockets, all_task_pockets


class PocketTests(unittest.TestCase):
    def test_transitive_boundary_and_center(self):
        result = candidate_pockets([[0,0,0],[6,0,0],[12,0,0],[30,0,0]], [.5,.8,.9,.99], .5, 6)
        cluster = next(p for p in result if len(p['residues']) == 3)
        self.assertEqual(cluster['residues'],[1,2,3])
        self.assertEqual(cluster['center_x'],6)
        self.assertAlmostEqual(cluster['score'], np.mean([.5,.8,.9])+.05*np.log1p(3))
        self.assertTrue(any(p['residues']==[4] for p in result))

    def test_tie_and_empty(self):
        self.assertEqual(candidate_pockets([[0,0,0]],[.1],.5,6),[])
        result=candidate_pockets([[0,0,0],[20,0,0]],[.9,.9],.5,6)
        self.assertEqual([p['residues'] for p in result],[[1],[2]])
        self.assertEqual([p['rank'] for p in result],[1,2])

    def test_invalid(self):
        for xyz,prob,threshold,cutoff in [([[0,0,float('nan')]],[.9],.5,6),
                ([[0,0,0]],[1.1],.5,6), ([[0,0,0]],[.9],.5,0),
                ([[0,0,0]],[.9],float('nan'),6), ([[0,0]],[.9],.5,6)]:
            with self.assertRaises(ValueError): candidate_pockets(xyz,prob,threshold,cutoff)

    def test_task_cutoffs_and_singletons(self):
        tasks=['metal','nucleic_acid','small_molecule']
        result=all_task_pockets([[0,0,0],[7,0,0]],{t:[.9,.9] for t in tasks},{t:.5 for t in tasks})
        self.assertEqual(len(result['nucleic_acid']),1)
        self.assertEqual(result['nucleic_acid'][0]['candidate_type'],'pocket')
        self.assertEqual(len(result['metal']),2)
        self.assertTrue(all(p['candidate_type']=='isolated_residue' for p in result['metal']))


if __name__=='__main__': unittest.main()
