import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from resbind_server.workflow import inspect_input, workspace_file

SEQ = 'MKWVTFISLLFLFSSAYSRGVFRR'


class WorkflowTests(unittest.TestCase):
    def test_valid_predict(self):
        result = inspect_input('>protein\n'+SEQ)
        self.assertEqual(result['length'],len(SEQ))
        self.assertFalse(result['labels_required'])
        self.assertEqual(result['missing_inputs'],['structure','netsurfp','prott5'])

    def test_normalization_reported(self):
        result = inspect_input('>protein\n'+SEQ+'BZUOJ')
        self.assertEqual(len(result['normalizations']),5)
        self.assertTrue(result['normalized_sequence'].endswith('XXXXX'))

    def test_invalid(self):
        for text in ['junk\n>p\n'+SEQ, '>\n'+SEQ, '>p\n', '>a\n'+SEQ+'\n>b\n'+SEQ,
                     '>p\n'+SEQ+'*', '>p\n'+SEQ+'1', '>p\n'+'A'*1025, '>p\nACGT'*10]:
            with self.subTest(text=text[:30]), self.assertRaises(ValueError):
                inspect_input(text)

    def test_labels_not_auto_split(self):
        for mode in ['evaluate','train']:
            result = inspect_input('>p\n'+SEQ,mode=mode)
            self.assertTrue(result['labels_required'])
            self.assertFalse(result['ready_for_supervised_execution'])

    def test_feature_mismatches(self):
        case = Path(__file__).resolve().parents[1] / 'examples/P07014'
        for keyword, filename in [('pdb','structure.pdb'), ('netsurfp','netsurfp.csv'), ('prott5','prott5.npz')]:
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                inspect_input('>different\n'+SEQ, **{keyword:case/filename})

    def test_length_and_mode_boundaries(self):
        for length in [20, 1024]:
            self.assertEqual(inspect_input('>p\n'+'M'*length)['length'],length)
        for length in [19, 1025]:
            with self.assertRaises(ValueError): inspect_input('>p\n'+'M'*length)
        with self.assertRaises(ValueError): inspect_input('>p\n'+SEQ,mode='unknown')

    def test_workspace(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'x.fasta').write_text('>p\n'+SEQ)
            self.assertEqual(workspace_file(root,'x.fasta',1000),root/'x.fasta')
            for value in ['../outside',str(root.parent/'outside')]:
                with self.assertRaises(ValueError): workspace_file(root,value,1000)
            with self.assertRaises(ValueError): workspace_file(root,'x.fasta',1)


if __name__=='__main__': unittest.main()
