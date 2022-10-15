# -*- coding: utf-8 -*-
import pathlib
import shutil

current_path = pathlib.Path(__file__)


def html_link(file, label, add_meta=True):
    label = label.replace(r"\ ", " ")
    label = label.replace(r"\rm", "")
    if add_meta:
        file = f"meta/{file}"
    return f"<li><a href={file}>{label}</a></li> \n"


def container_header(title):
    return f"""
        <div class='container'> \n
        <h2 id='{title}'> {title}</h2> \n
        <div class="three-fourth column markdown-body"> \n
    """


def html_figure(fig_name):
    return f"""
        <div class="figiterwrapper"> \n
        <figure> \n
        <img src="meta/{fig_name}.png"/> \n
        <figcaption aria-hidden="true"> \n
        <a href="meta/{fig_name}.png">.png</a> \n
        <a href="meta/{fig_name}.pdf">.pdf</a> \n
        </figcaption> \n
        </figure> \n
        </div> \n
    """


def write_html_container(title, figs=None, links=None, dataFrame=None):

    text = container_header(title)
    if dataFrame is not None:
        text += (
            "<div class='table'> \n"
            + dataFrame.to_html(justify="center", border=0)
            + "</div> \n"
        )

    if figs is not None:
        for fig_name in figs:
            text += html_figure(fig_name)

    if links is not None:
        for (file, label) in links:
            text += "</br> \n" + html_link(f"{file}.html", label)

    return text


def dump_html_index(html_report, html_index, report_path, report_title):
    """Dump report index to html.

    Parameters
    ----------
    html_report: str
        html report content
    html_index: str
        html index content
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
    text = text.replace("report_content", html_report)
    # add index
    text = text.replace("index_elements", html_index)
    with open(report_path.joinpath("index.html"), "w", encoding="utf-8") as f:
        f.write(text)
