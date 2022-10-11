# -*- coding: utf-8 -*-
import pathlib
import shutil
from ctypes import alignment

current_path = pathlib.Path(__file__)


def html_link(file, label):
    label = label.replace(r"\ ", " ")
    label = label.replace(r"\rm", "")
    return f"<li><a href=meta/{file}>{label}</a></li> \n"


def sub_index(title, index_list):
    html_index = f'<nav class="menu-docs-menu"> \n <a>{title}:</a> \n <ul>'
    for file, label in index_list:
        html_index += html_link(file, label)
    html_index += " </ul> </nav>"
    return html_index


def dump_html_index(fit_settings, html_index, report_path, report_title):
    """Dump report index to html.

    Parameters
    ----------
    fit_settings: pandas.DataFrame
        fit settings table
    html_index: str
        index content
    report_path: pathlib.Path
        report path
    report_title: str
        report title

    """
    index_html = current_path.parent.joinpath("assets/index.html")
    index_css = current_path.parent.joinpath("assets/index.css")
    shutil.copyfile(index_css, report_path.joinpath("index.css"))
    with open(index_html, "r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace(
        '<a class="masthead-logo">report_title</a>',
        f'<a class="masthead-logo">{report_title}</a>',
    )
    text = text.replace(
        "fit_settings", fit_settings.to_html(justify="center", border=0)
    )
    # add index
    text = text.replace("index_elements", html_index)
    with open(report_path.joinpath("index.html"), "w", encoding="utf-8") as f:
        f.write(text)
