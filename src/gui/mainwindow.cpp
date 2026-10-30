#include "mainwindow.hpp"

#include <algorithm>

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDoubleValidator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontDatabase>
#include <QFormLayout>
#include <QGridLayout>
#include <QGuiApplication>
#include <QScreen>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QIntValidator>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QScrollArea>
#include <QSettings>
#include <QSplitter>
#include <QStatusBar>
#include <QTextStream>
#include <QToolButton>
#include <QVBoxLayout>

using pagani_gui::Command;
using pagani_gui::Settings;

namespace {

QString tip(const QString& html) { return "<html><body><p>" + html + "</p></body></html>"; }

QWidget* row(std::initializer_list<QWidget*> widgets, bool stretch = true) {
    auto* w = new QWidget;
    auto* l = new QHBoxLayout(w);
    l->setContentsMargins(0, 0, 0, 0);
    for (QWidget* x : widgets) l->addWidget(x);
    if (stretch) l->addStretch(1);
    return w;
}

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QString("PAGANI Toolkit %1").arg(QApplication::applicationVersion()));
    setWindowIcon(QIcon(":/pagani/icon"));
    buildUi();
    QSettings st;
    Settings s;
    if (Settings::fromJson(st.value("lastSettings").toString(), s)) applySettings(s);
    scriptPath_->setText(st.value("lastScript").toString());
    restoreGeometry(st.value("geometry").toByteArray());
    updateEnabledState();
}

MainWindow::~MainWindow() = default;

QString MainWindow::appDir() const { return QCoreApplication::applicationDirPath(); }

void MainWindow::buildUi() {
    // ---------------------------------------------------------------- data
    auto* dataBox = new QGroupBox("Data");
    auto* dataGrid = new QGridLayout(dataBox);
    workingDir_ = new QLineEdit;
    workingDir_->setPlaceholderText("folder containing the preprocessed 4D fMRI images (.nii or .nii.gz)");
    workingDir_->setToolTip(tip("Folder with one preprocessed 4D NIfTI image (.nii or .nii.gz) per subject. Networks are "
                                "written to its <b>unweighted</b> or <b>weighted</b> sub-folder, and all metrics "
                                "are computed for every network (.csr) found there."));
    auto* wdButton = new QToolButton;
    wdButton->setText("...");
    connect(wdButton, &QToolButton::clicked, this, &MainWindow::browseWorkingDir);
    maskFile_ = new QLineEdit;
    maskFile_->setPlaceholderText("grey matter mask (.nii or .nii.gz) with the same dimensions as the images");
    auto* maskButton = new QToolButton;
    maskButton->setText("...");
    connect(maskButton, &QToolButton::clicked, this, &MainWindow::browseMask);
    maskThreshold_ = new QLineEdit("0.2");
    maskThreshold_->setValidator(new QDoubleValidator(this));
    maskThreshold_->setMaximumWidth(80);
    maskThreshold_->setToolTip(tip("Voxels whose mask value is &ge; this threshold are network nodes. "
                                   "Scaling factors stored in the mask header (e.g. SPM c1 maps) are applied."));
    networkType_ = new QComboBox;
    networkType_->addItems({"Unweighted", "Weighted"});
    networkType_->setToolTip(tip("Unweighted networks use the tools in <b>exefiles</b>, weighted networks "
                                 "(edge weight = correlation) those in <b>exefiles_weighted</b>."));
    dataGrid->addWidget(new QLabel("Working directory"), 0, 0);
    dataGrid->addWidget(workingDir_, 0, 1);
    dataGrid->addWidget(wdButton, 0, 2);
    dataGrid->addWidget(new QLabel("Mask file"), 1, 0);
    dataGrid->addWidget(maskFile_, 1, 1);
    dataGrid->addWidget(maskButton, 1, 2);
    dataGrid->addWidget(new QLabel("Mask threshold"), 2, 0);
    dataGrid->addWidget(row({maskThreshold_, new QLabel("   Network type"), networkType_}), 2, 1);

    // -------------------------------------------------------- construction
    construct_ = new QGroupBox("Network Construction (voxel-wise Pearson correlation on GPU)");
    construct_->setCheckable(true);
    construct_->setChecked(false);
    auto* cForm = new QGridLayout(construct_);
    standard_ = new QRadioButton("Standard");
    standard_->setChecked(true);
    standard_->setToolTip(tip("CUCorMat: supports group-level networks and saving the whole correlation matrix. "
                              "Needs host memory for the whole matrix when either option is used."));
    highRes_ = new QRadioButton("High resolution");
    highRes_->setToolTip(tip("CUCorMatHr: never stores the whole correlation matrix and can therefore build "
                             "networks with many more voxels; individual networks only."));
    auto* methodGroup = new QButtonGroup(this);
    methodGroup->addButton(standard_);
    methodGroup->addButton(highRes_);
    thresholdType_ = new QComboBox;
    thresholdType_->addItems({"Correlation (r >)", "Sparsity (%)"});
    thresholds_ = new QLineEdit;
    thresholds_->setPlaceholderText("e.g. 0.3 0.4   or   5 10 20");
    thresholds_->setToolTip(tip("One or more values separated by spaces. Correlation: edges with r greater than "
                                "the value. Sparsity: percentage of all voxel pairs kept as edges (strongest "
                                "correlations)."));
    group_ = new QCheckBox("Generate group-level network");
    ordinary_ = new QRadioButton("Ordinary average");
    ordinary_->setChecked(true);
    ordinary_->setToolTip(tip("Average of the correlation coefficients over subjects."));
    fisher_ = new QRadioButton("Fisher z average");
    fisher_->setToolTip(tip("Correlations are Fisher z-transformed before averaging and transformed back:"
                            "<br/><img src=':/pagani/fisher'/>"));
    auto* avgGroup = new QButtonGroup(this);
    avgGroup->addButton(ordinary_);
    avgGroup->addButton(fisher_);
    saveCormat_ = new QCheckBox("Save whole correlation matrix (.cormat)");
    cForm->addWidget(new QLabel("Method"), 0, 0);
    cForm->addWidget(row({standard_, highRes_}), 0, 1);
    cForm->addWidget(new QLabel("Threshold"), 1, 0);
    {
        auto* w = new QWidget;
        auto* l = new QHBoxLayout(w);
        l->setContentsMargins(0, 0, 0, 0);
        l->addWidget(thresholdType_);
        l->addWidget(thresholds_, 1);
        cForm->addWidget(w, 1, 1);
    }
    cForm->addWidget(row({group_, ordinary_, fisher_}), 2, 0, 1, 2);
    cForm->addWidget(saveCormat_, 3, 0, 1, 2);

    // -------------------------------------------------------- global metrics
    auto* globalBox = new QGroupBox("Global Metrics");
    auto* gGrid = new QGridLayout(globalBox);
    globalLp_ = new QCheckBox("Characteristic path length (Lp)");
    globalLp_->setToolTip(tip("Lp = 1 / global efficiency (harmonic mean of the shortest path lengths; distance "
                              "= 1/weight in weighted networks). Output: *_Lp.txt"));
    globalCp_ = new QCheckBox("Clustering coefficient (Cp)");
    globalCp_->setToolTip(tip("Mean clustering coefficient over all nodes. Output: *_cp.txt"));
    cpTypeLabel_ = new QLabel("Type");
    cpType_ = new QComboBox;
    cpType_->addItems({"Onnela", "Barrat"});
    cpType_->setToolTip(tip("Weighted clustering coefficient: Onnela et al. 2005 or Barrat et al. 2004."));
    gamma_ = new QCheckBox("Gamma (Cp / Cp_rand)");
    gamma_->setToolTip(tip("<img src=':/pagani/gamma'/><br/>Output: *_gamma.txt"));
    lambda_ = new QCheckBox("Lambda (Lp / Lp_rand)");
    lambda_->setToolTip(tip("<img src=':/pagani/lambda'/><br/>Output: *_lambda.txt"));
    sigma_ = new QCheckBox("Small-world sigma");
    sigma_->setToolTip(tip("<img src=':/pagani/sigma'/><br/>Requires gamma and lambda. Output: *_smallWorld.txt"));
    modularity_ = new QCheckBox("Modularity");
    modularity_->setToolTip(tip("Module partition (*_modu.nm, converted to NIfTI) and modularity Q of the network "
                                "and of the random networks (*_modu.txt)."));
    modAlgorithm_ = new QComboBox;
    modAlgorithm_->addItems({"Louvain", "Newman (GPU)", "Newman (CPU)"});
    randomNetworks_ = new QLineEdit("10");
    randomNetworks_->setValidator(new QIntValidator(0, 100000, this));
    randomNetworks_->setMaximumWidth(80);
    randomNetworks_->setToolTip(tip("Number of degree-preserving random networks (Maslov &amp; Sneppen) used for "
                                    "gamma, lambda, sigma and the modularity of random networks."));
    gGrid->addWidget(row({globalLp_, globalCp_, cpTypeLabel_, cpType_}), 0, 0);
    gGrid->addWidget(row({gamma_, lambda_, sigma_}), 1, 0);
    gGrid->addWidget(row({modularity_, modAlgorithm_, new QLabel("   Random networks"), randomNetworks_}), 2, 0);

    // --------------------------------------------------------- nodal metrics
    auto* nodalBox = new QGroupBox("Nodal Metrics (converted to NIfTI images with the mask)");
    auto* nGrid = new QGridLayout(nodalBox);
    nodalEff_ = new QCheckBox("Nodal efficiency");
    nodalCp_ = new QCheckBox("Clustering coefficient");
    degree_ = new QCheckBox("Degree centrality");
    degree_->setToolTip(tip("Unweighted: number of edges. Weighted: nodal strength."));
    betweenness_ = new QCheckBox("Betweenness centrality");
    eigenvector_ = new QCheckBox("Eigenvector centrality");
    participation_ = new QCheckBox("Participation coefficient");
    participation_->setToolTip(tip("Requires the module partition (Modularity is enabled automatically)."));
    nGrid->addWidget(nodalEff_, 0, 0);
    nGrid->addWidget(nodalCp_, 0, 1);
    nGrid->addWidget(degree_, 0, 2);
    nGrid->addWidget(betweenness_, 1, 0);
    nGrid->addWidget(eigenvector_, 1, 1);
    nGrid->addWidget(participation_, 1, 2);

    // ------------------------------------------------------------- actions
    scriptPath_ = new QLineEdit;
    scriptPath_->setPlaceholderText("script file (.bat)");
    saveButton_ = new QPushButton("Save Script...");
    loadButton_ = new QPushButton("Load Script...");
    runButton_ = new QPushButton("Run");
    runButton_->setDefault(true);
    runButton_->setMinimumWidth(110);
    stopButton_ = new QPushButton("Stop");
    stopButton_->setEnabled(false);
    connect(saveButton_, &QPushButton::clicked, this, &MainWindow::saveScript);
    connect(loadButton_, &QPushButton::clicked, this, &MainWindow::loadScript);
    connect(runButton_, &QPushButton::clicked, this, &MainWindow::run);
    connect(stopButton_, &QPushButton::clicked, this, &MainWindow::stop);
    auto* scriptRow = new QHBoxLayout;
    scriptRow->addWidget(new QLabel("Script"));
    scriptRow->addWidget(scriptPath_, 1);
    scriptRow->addWidget(loadButton_);
    scriptRow->addWidget(saveButton_);
    auto* runRow = new QHBoxLayout;
    runRow->addStretch(1);
    runRow->addWidget(stopButton_);
    runRow->addWidget(runButton_);

    settingsPanel_ = new QWidget;
    auto* left = new QVBoxLayout(settingsPanel_);
    left->addWidget(dataBox);
    left->addWidget(construct_);
    left->addWidget(globalBox);
    left->addWidget(nodalBox);
    left->addStretch(1);
    auto* scroll = new QScrollArea;
    scroll->setWidget(settingsPanel_);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setMinimumWidth(settingsPanel_->sizeHint().width() + 24);

    log_ = new QPlainTextEdit;
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(200000);
    log_->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    log_->setPlaceholderText("Messages of the running tools appear here.");
    progress_ = new QProgressBar;
    progress_->setRange(0, 1);
    progress_->setValue(0);
    progress_->setFormat("");
    status_ = new QLabel("Ready");
    auto* right = new QWidget;
    auto* rl = new QVBoxLayout(right);
    rl->addWidget(new QLabel("Log"));
    rl->addWidget(log_, 1);
    rl->addWidget(progress_);
    rl->addWidget(status_);

    auto* split = new QSplitter;
    split->addWidget(scroll);
    split->addWidget(right);
    split->setStretchFactor(0, 0);
    split->setStretchFactor(1, 1);
    // initial size: all settings visible without scrolling when the screen allows it
    const QRect avail = QGuiApplication::primaryScreen()->availableGeometry();
    const int w0 = std::min(scroll->minimumWidth() + 560, static_cast<int>(avail.width() * 0.92));
    const int h0 = std::min(settingsPanel_->sizeHint().height() + 150, static_cast<int>(avail.height() * 0.9));
    split->setSizes({scroll->minimumWidth(), std::max(300, w0 - scroll->minimumWidth())});

    auto* central = new QWidget;
    auto* cl = new QVBoxLayout(central);
    cl->addWidget(split, 1);
    cl->addLayout(scriptRow);
    cl->addLayout(runRow);
    setCentralWidget(central);
    resize(w0, h0);

    // dependencies between options
    for (QCheckBox* c : {group_, saveCormat_, globalLp_, globalCp_, gamma_, lambda_, sigma_, modularity_, nodalEff_,
                         nodalCp_, degree_, betweenness_, eigenvector_, participation_})
        connect(c, &QCheckBox::toggled, this, &MainWindow::updateEnabledState);
    connect(construct_, &QGroupBox::toggled, this, &MainWindow::updateEnabledState);
    connect(standard_, &QRadioButton::toggled, this, &MainWindow::updateEnabledState);
    connect(networkType_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::updateEnabledState);
    connect(sigma_, &QCheckBox::toggled, this, [this](bool on) {
        if (on) { gamma_->setChecked(true); lambda_->setChecked(true); }
    });
    connect(gamma_, &QCheckBox::toggled, this, [this](bool on) { if (!on) sigma_->setChecked(false); });
    connect(lambda_, &QCheckBox::toggled, this, [this](bool on) { if (!on) sigma_->setChecked(false); });
    connect(participation_, &QCheckBox::toggled, this, [this](bool on) { if (on) modularity_->setChecked(true); });
    connect(modularity_, &QCheckBox::toggled, this, [this](bool on) { if (!on) participation_->setChecked(false); });
}

void MainWindow::updateEnabledState() {
    const bool weighted = networkType_->currentIndex() == 1;
    const bool standard = standard_->isChecked() && construct_->isChecked();
    group_->setEnabled(standard);
    saveCormat_->setEnabled(standard);
    ordinary_->setEnabled(standard && group_->isChecked());
    fisher_->setEnabled(standard && group_->isChecked());
    cpType_->setEnabled(weighted && (globalCp_->isChecked() || nodalCp_->isChecked() || gamma_->isChecked()));
    cpTypeLabel_->setEnabled(cpType_->isEnabled());
    cpType_->setVisible(weighted);
    cpTypeLabel_->setVisible(weighted);
    modAlgorithm_->setEnabled(modularity_->isChecked());
    if (weighted && betweenness_->isChecked()) betweenness_->setChecked(false);
    betweenness_->setEnabled(!weighted);
    betweenness_->setToolTip(weighted ? tip("Betweenness centrality is only available for unweighted networks.")
                                      : tip("Brandes' algorithm on the GPU; output *_bc.nm"));
    const bool needRandom = gamma_->isChecked() || lambda_->isChecked() || sigma_->isChecked() || modularity_->isChecked();
    randomNetworks_->setEnabled(needRandom);
}

Settings MainWindow::settings() const {
    Settings s;
    s.workingDir = workingDir_->text().trimmed();
    s.maskFile = maskFile_->text().trimmed();
    s.maskThreshold = maskThreshold_->text().trimmed();
    s.weighted = networkType_->currentIndex() == 1;
    s.construct = construct_->isChecked();
    s.highResolution = highRes_->isChecked();
    s.sparsityThreshold = thresholdType_->currentIndex() == 1;
    s.thresholds = thresholds_->text().trimmed();
    s.groupNetwork = group_->isChecked() && !s.highResolution;
    s.fisherAverage = fisher_->isChecked();
    s.saveCorMatrix = saveCormat_->isChecked() && !s.highResolution;
    s.globalLp = globalLp_->isChecked();
    s.globalCp = globalCp_->isChecked();
    s.gamma = gamma_->isChecked();
    s.lambda = lambda_->isChecked();
    s.sigma = sigma_->isChecked();
    s.modularity = modularity_->isChecked();
    s.modularityAlgorithm = modAlgorithm_->currentIndex();
    s.cpType = cpType_->currentIndex() == 1 ? 1 : 2;
    s.randomNetworks = randomNetworks_->text().trimmed();
    s.nodalEfficiency = nodalEff_->isChecked();
    s.betweenness = betweenness_->isChecked() && !s.weighted;
    s.nodalCp = nodalCp_->isChecked();
    s.degree = degree_->isChecked();
    s.eigenvector = eigenvector_->isChecked();
    s.participation = participation_->isChecked();
    return s;
}

void MainWindow::applySettings(const Settings& s) {
    workingDir_->setText(QDir::toNativeSeparators(s.workingDir));
    maskFile_->setText(QDir::toNativeSeparators(s.maskFile));
    maskThreshold_->setText(s.maskThreshold);
    networkType_->setCurrentIndex(s.weighted ? 1 : 0);
    construct_->setChecked(s.construct);
    (s.highResolution ? highRes_ : standard_)->setChecked(true);
    thresholdType_->setCurrentIndex(s.sparsityThreshold ? 1 : 0);
    thresholds_->setText(s.thresholds);
    group_->setChecked(s.groupNetwork);
    (s.fisherAverage ? fisher_ : ordinary_)->setChecked(true);
    saveCormat_->setChecked(s.saveCorMatrix);
    globalLp_->setChecked(s.globalLp);
    globalCp_->setChecked(s.globalCp);
    gamma_->setChecked(s.gamma);
    lambda_->setChecked(s.lambda);
    sigma_->setChecked(s.sigma);
    modularity_->setChecked(s.modularity);
    modAlgorithm_->setCurrentIndex(qBound(0, s.modularityAlgorithm, 2));
    cpType_->setCurrentIndex(s.cpType == 1 ? 1 : 0);
    randomNetworks_->setText(s.randomNetworks);
    nodalEff_->setChecked(s.nodalEfficiency);
    betweenness_->setChecked(s.betweenness);
    nodalCp_->setChecked(s.nodalCp);
    degree_->setChecked(s.degree);
    eigenvector_->setChecked(s.eigenvector);
    participation_->setChecked(s.participation);
    updateEnabledState();
}

void MainWindow::browseWorkingDir() {
    QString d = QFileDialog::getExistingDirectory(this, "Working directory", workingDir_->text());
    if (!d.isEmpty()) workingDir_->setText(QDir::toNativeSeparators(d));
}

void MainWindow::browseMask() {
    QString start = maskFile_->text().isEmpty() ? workingDir_->text() : maskFile_->text();
    QString f = QFileDialog::getOpenFileName(this, "Mask file", start, "NIfTI images (*.nii *.nii.gz);;All files (*)");
    if (!f.isEmpty()) maskFile_->setText(QDir::toNativeSeparators(f));
}

bool MainWindow::checkSettings(const Settings& s) {
    const QStringList errors = pagani_gui::validate(s);
    if (errors.isEmpty()) return true;
    QMessageBox::warning(this, "Please check the settings", "<ul><li>" + errors.join("</li><li>") + "</li></ul>");
    return false;
}

void MainWindow::saveScript() {
    const Settings s = settings();
    if (!checkSettings(s)) return;
    QString start = scriptPath_->text().isEmpty() ? QDir(s.workingDir).filePath("pagani_script.bat") : scriptPath_->text();
    QString f = QFileDialog::getSaveFileName(this, "Save script", start, "Batch script (*.bat);;All files (*)");
    if (f.isEmpty()) return;
    QFile file(f);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        QMessageBox::critical(this, "Save script", "Cannot write " + f);
        return;
    }
    file.write(pagani_gui::buildScript(s, appDir(), pagani_gui::newRandomSeed()).toLocal8Bit());
    scriptPath_->setText(QDir::toNativeSeparators(f));
    status_->setText("Script saved: " + QDir::toNativeSeparators(f));
}

void MainWindow::loadScript() {
    QString f = QFileDialog::getOpenFileName(this, "Load script", scriptPath_->text(), "Batch script (*.bat *.sh);;All files (*)");
    if (f.isEmpty()) return;
    QFile file(f);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "Load script", "Cannot read " + f);
        return;
    }
    const QString text = QString::fromLocal8Bit(file.readAll());
    Settings s;
    QString msg;
    if (!pagani_gui::parseScript(text, s, &msg)) {
        QMessageBox::warning(this, "Load script", msg);
        return;
    }
    applySettings(s);
    scriptPath_->setText(QDir::toNativeSeparators(f));
    status_->setText(msg.section('\n', 0, 0));
    if (msg.contains('\n')) QMessageBox::information(this, "Load script", msg);
}

void MainWindow::run() {
    const Settings s = settings();
    if (!checkSettings(s)) return;
    queue_ = pagani_gui::buildCommands(s, appDir());
    for (const Command& c : queue_) {
        if (!QFileInfo(c.program).isFile()) {
            QMessageBox::critical(this, "Missing program",
                                  "Cannot find " + c.program + "\n\nThe exefiles and exefiles_weighted folders must be "
                                  "next to PAGANI.exe.");
            return;
        }
    }
    QSettings().setValue("lastSettings", s.toJson());
    log_->clear();
    appendLog(QString("PAGANI Toolkit %1 - run started %2")
                  .arg(QApplication::applicationVersion(), QDateTime::currentDateTime().toString(Qt::ISODate)),
              "#1c3a5e");
    // one seed for every step, so that all metrics use the same random networks
    env_ = QProcessEnvironment::systemEnvironment();
    if (pagani_gui::usesRandomNetworks(s)) {
        if (!env_.contains("PAGANI_SEED")) env_.insert("PAGANI_SEED", pagani_gui::newRandomSeed());
        appendLog("Random networks: PAGANI_SEED = " + env_.value("PAGANI_SEED") +
                      " (lambda, gamma, sigma and Q use the same random networks; set PAGANI_SEED to this "
                      "value to reproduce them)",
                  "#666666");
    }
    progress_->setRange(0, queue_.size());
    progress_->setFormat("%v / %m steps");
    progress_->setValue(0);
    current_ = -1;
    stopping_ = false;
    setRunning(true);
    startNext();
}

void MainWindow::startNext() {
    ++current_;
    if (current_ >= queue_.size()) {
        finishRun(true, "All steps finished.");
        return;
    }
    const Command& c = queue_[current_];
    QStringList shown;
    shown << c.program;
    for (const QString& a : c.args) shown << (a.contains(' ') ? '"' + a + '"' : a);
    appendLog(QString("\n==== [%1/%2] %3").arg(current_ + 1).arg(queue_.size()).arg(c.title), "#1c3a5e");
    appendLog(shown.join(' '), "#666666");
    status_->setText(QString("Running step %1 of %2: %3").arg(current_ + 1).arg(queue_.size()).arg(c.title));
    process_ = new QProcess(this);
    process_->setProcessChannelMode(QProcess::MergedChannels);
    process_->setWorkingDirectory(QFileInfo(c.program).absolutePath());
    process_->setProcessEnvironment(env_);
    connect(process_, &QProcess::readyReadStandardOutput, this, &MainWindow::processOutput);
    connect(process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &MainWindow::processFinished);
    connect(process_, &QProcess::errorOccurred, this, &MainWindow::processError);
    process_->start(c.program, c.args);
}

void MainWindow::processOutput() {
    if (!process_) return;
    QString text = QString::fromLocal8Bit(process_->readAllStandardOutput());
    text.replace("\r\n", "\n");
    if (text.endsWith('\n')) text.chop(1);
    if (!text.isEmpty()) appendLog(text);
}

void MainWindow::processFinished(int exitCode, QProcess::ExitStatus status) {
    processOutput();
    process_->deleteLater();
    process_ = nullptr;
    if (stopping_) {
        finishRun(false, "Stopped by the user.");
        return;
    }
    if (status != QProcess::NormalExit) {
        finishRun(false, QString("Step %1 crashed.").arg(current_ + 1));
        return;
    }
    if (exitCode != 0) {
        finishRun(false, QString("Step %1 (%2) failed with exit code %3. See the log for details.")
                             .arg(current_ + 1).arg(queue_[current_].title).arg(exitCode));
        return;
    }
    progress_->setValue(current_ + 1);
    startNext();
}

void MainWindow::processError(QProcess::ProcessError error) {
    if (error == QProcess::FailedToStart && process_) {
        appendLog("Cannot start " + queue_[current_].program + ": " + process_->errorString(), "#b00020");
        process_->deleteLater();
        process_ = nullptr;
        finishRun(false, "A program could not be started.");
    }
}

void MainWindow::stop() {
    if (!process_) return;
    stopping_ = true;
    appendLog("Stopping...", "#b00020");
    process_->kill();
}

void MainWindow::finishRun(bool ok, const QString& message) {
    setRunning(false);
    appendLog("\n" + message, ok ? "#1b7f3b" : "#b00020");
    status_->setText(message);
    if (ok) {
        progress_->setValue(progress_->maximum());
        QMessageBox::information(this, "PAGANI", message + "\n\nResults are in " + pagani_gui::networkDirectory(settings()));
    } else if (!stopping_) {
        QMessageBox::warning(this, "PAGANI", message);
    }
}

void MainWindow::setRunning(bool running) {
    settingsPanel_->setEnabled(!running);
    runButton_->setEnabled(!running);
    saveButton_->setEnabled(!running);
    loadButton_->setEnabled(!running);
    stopButton_->setEnabled(running);
}

void MainWindow::appendLog(const QString& text, const QString& color) {
    if (color.isEmpty()) {
        log_->appendPlainText(text);
    } else {
        QString html = text.toHtmlEscaped();
        html.replace("\n", "<br/>");
        log_->appendHtml("<span style='color:" + color + "'>" + html + "</span>");
    }
}

void MainWindow::closeEvent(QCloseEvent* e) {
    if (process_) {
        if (QMessageBox::question(this, "PAGANI", "A computation is running. Stop it and quit?") != QMessageBox::Yes) {
            e->ignore();
            return;
        }
        stopping_ = true;
        process_->kill();
        process_->waitForFinished(3000);
    }
    QSettings st;
    st.setValue("lastSettings", settings().toJson());
    st.setValue("lastScript", scriptPath_->text());
    st.setValue("geometry", saveGeometry());
    e->accept();
}
