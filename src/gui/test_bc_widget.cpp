// Exercise the actual settings widgets that used to clear weighted BC.
#include "mainwindow.hpp"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QSettings>
#include <QTemporaryDir>

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QTemporaryDir dir;
    if (!dir.isValid()) return 1;
    app.setOrganizationName("PAGANI tests");
    app.setApplicationName("weighted-bc");
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, dir.path());
    MainWindow window;
    auto* type = window.findChild<QComboBox*>("networkType");
    auto* bc = window.findChild<QCheckBox*>("betweenness");
    auto* backend = window.findChild<QComboBox*>("bcBackend");
    if (!type || !bc || !backend) return 2;
    bc->setChecked(true);
    type->setCurrentIndex(1);
    if (!bc->isChecked() || !bc->isEnabled() || backend->isHidden() || !backend->isEnabled()) return 3;
    backend->setCurrentIndex(backend->findData("gpu"));
    type->setCurrentIndex(0);
    if (!backend->isHidden() || !bc->isChecked()) return 4;
    type->setCurrentIndex(1);
    if (backend->currentData().toString() != "gpu") return 5;
    bc->setChecked(false);
    return backend->isEnabled() ? 6 : 0;
}
