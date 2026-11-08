"""Report integrity regressions; no inference or Score recomputation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.audit_gcg_negative_energy import verify_hashed_files,verify_report_bundle
from scripts.report_gcg_negative_energy import report
from utils.artifact_io import sha256_file


class ReportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def write(self,name,value):
        path=self.root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value,dict) else value)
        return path

    def test_audited_file_checks_success_missing_and_changed(self):
        file=self.write('input.csv','metric,value\nE_neg,1\n')
        expected={'input.csv':sha256_file(file)}
        self.assertEqual(verify_hashed_files(self.root,expected),expected)
        with self.assertRaisesRegex(ValueError,'missing audited hash: missing.csv'):
            verify_hashed_files(self.root,expected,['missing.csv'])
        file.write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch: .*input.csv'):
            verify_hashed_files(self.root,expected)
        file.unlink()
        with self.assertRaisesRegex(ValueError,'missing hashed file'):
            verify_hashed_files(self.root,expected)

    def test_report_rejects_changed_csv_before_read_or_output(self):
        file=self.write('tables/head_distributions.csv','original')
        self.write('output_hashes.json',{'tables/head_distributions.csv':sha256_file(file)})
        file.write_text('modified')
        target=self.root/'revision'
        with patch('scripts.report_gcg_negative_energy.pd.read_csv') as read:
            with self.assertRaisesRegex(ValueError,'hash mismatch: .*head_distributions.csv'):
                report(self.root,target)
            read.assert_not_called()
        self.assertFalse(target.exists())

    def bundle(self):
        source=self.write('source.py','# reporting source')
        csv=self.write('input.csv','value\n1\n')
        ledger=self.write('audited.json',{'input.csv':sha256_file(csv)})
        for name in ('findings.md','MTG_findings.md','README.md','manifest.json'):
            self.write(name,'{}' if name.endswith('.json') else '# report')
        self.write('report_manifest.json',dict(input_hashes={str(csv):sha256_file(csv)},
            source_hashes={str(source):sha256_file(source)},audited_ledger=str(ledger),
            audited_ledger_sha256=sha256_file(ledger)))
        names=('findings.md','MTG_findings.md','README.md','manifest.json','report_manifest.json')
        self.write('report_hashes.json',{n:sha256_file(self.root/n) for n in names})

    def test_detached_bundle_success_and_report_manifest_corruption(self):
        self.bundle()
        self.assertEqual(verify_report_bundle(self.root)['report_files_verified'],5)
        hashes=json.loads((self.root/'report_hashes.json').read_text())
        self.assertNotIn('report_hashes.json',hashes)
        for name in ('findings.md','manifest.json','report_manifest.json'):
            original=(self.root/name).read_bytes()
            (self.root/name).write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch: .*'+name):
                verify_report_bundle(self.root)
            (self.root/name).write_bytes(original)

    def test_bundle_rejects_input_source_ledger_or_inventory_change(self):
        self.bundle()
        for name in ('input.csv','source.py','audited.json'):
            original=(self.root/name).read_bytes()
            (self.root/name).write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                verify_report_bundle(self.root)
            (self.root/name).write_bytes(original)
        self.write('report_hashes.json',{})
        with self.assertRaisesRegex(ValueError,'inventory mismatch'):
            verify_report_bundle(self.root)


if __name__=='__main__':unittest.main()
