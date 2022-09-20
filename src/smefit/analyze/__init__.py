# -*- coding: utf-8 -*-
import subprocess

import yaml
from PyPDF2 import PdfFileReader, PdfFileWriter

from ..log import logging
from .report import Report

_logger = logging.getLogger(__name__)


def run_report(root_path, report_config):
    """
    Run the analysis given a report card name

    Parameters
    ----------
        report_config : str
            report configuration dictionary name
    """
    with open(f"{root_path}/analyze/{report_config}.yaml", encoding="utf-8") as f:
        report_config = yaml.safe_load(f)

    _logger.info(f"Analysing : {report_config['result_IDs']}")

    report_name = report_config["name"]
    report_path = report_config["report_path"]
    dir_path = f"{report_path}/{report_name}"

    # Clean output folder if exists
    try:
        subprocess.call(f"rm -rf {dir_path}", shell=True)
    except FileNotFoundError:
        pass

    subprocess.call(f"mkdir -p {report_path}", shell=True)
    subprocess.call(f"mkdir -p {dir_path}", shell=True)

    # Initialize ANALYZE class
    report = Report(report_path, report_config["result_path"], report_config)

    # Things to include in report
    if report_config["summary"]:
        report.summary()

    if "coefficients_plots" in report_config:
        report.coefficients(**report_config["coefficients_plots"])

    if "correlations" in report_config:
        report.correlations(**report_config["correlations"])

    # Combine PDF files together into raw pdf report
    subprocess.call(
        f"gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite \
                -sOutputFile={dir_path}/report_{report_name}_raw.pdf `ls -rt {dir_path}/*.pdf`",
        shell=True,
    )
    subprocess.call(f"mkdir -p {dir_path}/meta", shell=True)
    subprocess.call(f"mv {dir_path}/*.* {dir_path}/meta/.", shell=True)
    subprocess.call(f"mv {dir_path}/meta/report_*.pdf  {dir_path}/", shell=True)

    # TODO:
    # 1) add an index,
    # 2) run latex only a the end
    # 3) remove rotation
    # Rotate PDF pages if necessary and create final report
    with open(f"{dir_path}/report_{report_name}_raw.pdf", "rb") as pdf_in:
        pdf_reader = PdfFileReader(pdf_in)
        pdf_writer = PdfFileWriter()
        for pagenum in range(pdf_reader.numPages):
            pdfpage = pdf_reader.getPage(pagenum)
            orientation = pdfpage.get("/Rotate")
            if orientation == 90:
                pdfpage.rotateCounterClockwise(90)
            pdf_writer.addPage(pdfpage)
        with open(f"{dir_path}/report_{report_name}.pdf", "wb") as pdf_out:
            pdf_writer.write(pdf_out)
        pdf_out.close()
    pdf_in.close()

    # Remove old (raw) PDF file
    subprocess.call(f"rm {dir_path}/report_{report_name}_raw.pdf", shell=True)
