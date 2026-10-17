#include <QApplication>

#include "mainwindow.hpp"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QApplication::setOrganizationName("PAGANI");
    QApplication::setApplicationName("PAGANI Toolkit");
    QApplication::setApplicationVersion(PAGANI_VERSION);
    MainWindow w;
    w.show();
    return app.exec();
}
