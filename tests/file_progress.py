"""Per-file terminal progress; pytest still owns result accounting and failure summaries."""
from collections import OrderedDict

import pytest
from _pytest.terminal import TerminalReporter
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


def pytest_addoption(parser):
    parser.addoption('--file-progress', choices=('auto', 'on', 'off'), default='auto',
                     help='Live per-file progress (auto: interactive, non-verbose runs only).')


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    if hasattr(config, 'workerinput') or config.option.collectonly:
        return
    reporter = config.pluginmanager.getplugin('terminalreporter')
    mode = config.getoption('--file-progress')
    if reporter is None or mode == 'off':
        return
    output = reporter._tw._file
    if mode == 'auto' and (not output.isatty() or Console(file=output).is_dumb_terminal
                           or config.option.verbose > 0 or config.option.capture == 'no'):
        return
    config.pluginmanager.unregister(reporter)
    config.pluginmanager.register(FileProgressReporter(config, output), 'terminalreporter')


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.getplugin('terminalreporter')
    if isinstance(reporter, FileProgressReporter):
        reporter.stop_progress()


def pytest_unconfigure(config):
    reporter = config.pluginmanager.getplugin('terminalreporter')
    if isinstance(reporter, FileProgressReporter):
        reporter.stop_progress()


class FileProgressReporter(TerminalReporter):
    def __init__(self, config, output):
        super().__init__(config, output)
        self._show_progress_info = False
        self.files = OrderedDict()
        self.outcomes = {}
        self.finished = set()
        self.live = None
        self.console = Console(file=output, force_terminal=True)

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionstart(self, session):
        # Print the header before xdist starts writing its worker startup messages.
        super().pytest_sessionstart(session)

    def add_items(self, nodeids):
        for nodeid in nodeids:
            self.files.setdefault(nodeid.split('::')[0], set()).add(nodeid)

    def pytest_collection_finish(self, session):
        super().pytest_collection_finish(session)
        self.add_items(item.nodeid for item in session.items)

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        self.add_items(ids)

    def pytest_runtest_logstart(self, nodeid, location):
        self.add_items([nodeid])
        self.refresh_progress()

    def pytest_runtest_logreport(self, report):
        # Preserve pytest's accounting while replacing only its per-report terminal output.
        self._tests_ran = True
        category, letter, word = self.config.hook.pytest_report_teststatus(report=report, config=self.config)
        self._add_stats(category, [report])
        self.add_items([report.nodeid])
        if letter or word:
            self._progress_nodeids_reported.add(report.nodeid)
            previous = self.outcomes.get(report.nodeid)
            if previous not in ('failed', 'error'):
                self.outcomes[report.nodeid] = category
        if report.when == 'teardown':
            self.finished.add(report.nodeid)
        # xdist creates a synthetic report when a worker crashes, without a teardown report.
        if report.when not in ('setup', 'call', 'teardown'):
            self.finished.add(report.nodeid)
        self.refresh_progress()

    def refresh_progress(self):
        table = Table.grid(padding=(0, 2))
        table.add_column(no_wrap=True, overflow='ellipsis')
        table.add_column(no_wrap=True)
        for filename, nodeids in self.files.items():
            done = len(nodeids & self.finished)
            failed = sum(self.outcomes.get(nodeid) in ('failed', 'error') for nodeid in nodeids)
            skipped = sum(self.outcomes.get(nodeid) in ('skipped', 'xfailed') for nodeid in nodeids)
            filled = 12 * done // len(nodeids)
            status = Text('[' + '=' * filled + ' ' * (12 - filled) + f'] {done}/{len(nodeids)}',
                          style='red' if failed else 'green' if done == len(nodeids) else 'cyan')
            if failed:
                status.append(f'  F:{failed}', style='bold red')
            if skipped:
                status.append(f'  s/x:{skipped}', style='yellow')
            table.add_row(Text(filename), status)
        if self.live is None:
            self.ensure_newline()
            self.live = Live(table, console=self.console, auto_refresh=False,
                             redirect_stdout=False, redirect_stderr=False)
            self.live.start()
        self.live.update(table, refresh=True)

    def stop_progress(self):
        if self.live is not None:
            self.live.stop()
            self.live = None
