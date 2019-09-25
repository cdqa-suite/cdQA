import os
import wget
import pytest
from cdqa.utils.converters import pdf_converter, md_converter


@pytest.fixture(scope="session")
def download_test_assets(tmpdir_factory):
    assets_urls = [
        # PDF
        "https://invest.bnpparibas.com/documents/1q19-pr-12648",
        "https://invest.bnpparibas.com/documents/4q18-pr-18000",
        "https://invest.bnpparibas.com/documents/4q17-pr",
        # MD
        "https://raw.githubusercontent.com/cdqa-suite/cdQA/master/README.md",
        "https://raw.githubusercontent.com/huggingface/pytorch-transformers/master/docs/source/quickstart.md",
        "https://raw.githubusercontent.com/huggingface/pytorch-transformers/master/docs/source/migration.md",
    ]

    print("\nDownloading assets...")
    fn = tmpdir_factory.mktemp("assets_data")
    for url in assets_urls:
        wget.download(url=url, out=str(fn))
    return fn


class Test_converter:
    @pytest.fixture(autouse=True)
    def get_assets_folder(self, download_test_assets):
        self.assets_folder = download_test_assets

    def df_converter_check(self, df, include_line_breaks=False):
        errors = []
        # replace assertions by conditions
        if not df.shape == (3, 2):
            errors.append("resulting dataframe has unexpected shape.")
        if not (isinstance(df.paragraphs[0][0], str) and isinstance(df.title[0], str)):
            errors.append("paragraph column content has wrong format.")
        if include_line_breaks:
            para_len = [len(df.paragraphs[i]) for i in range(df.shape[0])]
            para_len.sort()
            if not para_len == [144, 220, 265]:
                errors.append(f"error in number of paragraphs : {para_len}")

        # assert no error message has been registered, else print messages
        assert not errors, "errors occured:\n{}".format("\n".join(errors))

    def test_md_converter(self):
        df = md_converter(directory_path=self.assets_folder)
        self.df_converter_check(df)

    def test_pdf_converter(self):
        df = pdf_converter(directory_path=self.assets_folder)
        self.df_converter_check(df)
        df_line_para = pdf_converter(
            directory_path=self.assets_folder, include_line_breaks=True
        )
        self.df_converter_check(df_line_para, True)
