#pragma once

#include <QMainWindow>
#include <QProcess>
#include <QVector>

#include "pipeline.hpp"

class QCheckBox;
class QComboBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QRadioButton;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent* e) override;

private slots:
    void browseWorkingDir();
    void browseMask();
    void saveScript();
    void loadScript();
    void run();
    void stop();
    void updateEnabledState();
    void processOutput();
    void processFinished(int exitCode, QProcess::ExitStatus status);
    void processError(QProcess::ProcessError error);

private:
    void buildUi();
    pagani_gui::Settings settings() const;
    void applySettings(const pagani_gui::Settings& s);
    bool checkSettings(const pagani_gui::Settings& s);
    void startNext();
    void finishRun(bool ok, const QString& message);
    void appendLog(const QString& text, const QString& color = QString());
    void setRunning(bool running);
    QString appDir() const;

    // data
    QLineEdit* workingDir_;
    QLineEdit* maskFile_;
    QLineEdit* maskThreshold_;
    QComboBox* networkType_;
    // construction
    QGroupBox* construct_;
    QRadioButton* standard_;
    QRadioButton* highRes_;
    QComboBox* thresholdType_;
    QLineEdit* thresholds_;
    QCheckBox* group_;
    QRadioButton* ordinary_;
    QRadioButton* fisher_;
    QCheckBox* saveCormat_;
    // global
    QCheckBox* globalLp_;
    QCheckBox* globalCp_;
    QComboBox* cpType_;
    QLabel* cpTypeLabel_;
    QCheckBox* gamma_;
    QCheckBox* lambda_;
    QCheckBox* sigma_;
    QCheckBox* modularity_;
    QComboBox* modAlgorithm_;
    QLineEdit* randomNetworks_;
    // nodal
    QCheckBox* nodalEff_;
    QCheckBox* nodalCp_;
    QCheckBox* degree_;
    QCheckBox* betweenness_;
    QCheckBox* eigenvector_;
    QCheckBox* participation_;
    // run
    QLineEdit* scriptPath_;
    QPushButton* runButton_;
    QPushButton* stopButton_;
    QPushButton* saveButton_;
    QPushButton* loadButton_;
    QPlainTextEdit* log_;
    QProgressBar* progress_;
    QLabel* status_;
    QWidget* settingsPanel_;

    QProcess* process_ = nullptr;
    QVector<pagani_gui::Command> queue_;
    int current_ = -1;
    bool stopping_ = false;
};
