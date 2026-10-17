#include <QApplication>

#include "mainwindow.hpp"

int main(int argc, char* argv[]) {
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    // scale the layout with the display scaling (Qt 6 does this by default);
    // without it only the fonts grow and the window is cramped on 150 % displays
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
    QGuiApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);
#endif
    QApplication app(argc, argv);
    QApplication::setOrganizationName("PAGANI");
    QApplication::setApplicationName("PAGANI Toolkit");
    QApplication::setApplicationVersion(PAGANI_VERSION);
    MainWindow w;
    w.show();
    return app.exec();
}
