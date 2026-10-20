// Unit tests for the GUI command generation and script round trip.
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QTemporaryDir>
#include <cstdio>
#include <cstdlib>
#include <QFileInfo>

#include "pipeline.hpp"

using namespace pagani_gui;

static int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                             \
        }                                                                           \
    } while (0)

static bool same(const Settings& a, const Settings& b) { return a.toJson() == b.toJson(); }

static const Command* find(const QVector<Command>& cmds, const QString& exe) {
    for (const auto& c : cmds)
        if (QFileInfo(c.program).baseName() == exe) return &c;
    return nullptr;
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    // helper mode for the Python end-to-end test: print the script for a settings file
    if (argc == 4 && QString(argv[1]) == "--script") {
        QFile f(argv[2]);
        Settings s;
        if (!f.open(QIODevice::ReadOnly) || !Settings::fromJson(QString::fromUtf8(f.readAll()), s)) return 2;
        const QStringList errors = validate(s);
        if (!errors.isEmpty()) {
            std::fprintf(stderr, "%s\n", errors.join("\n").toLocal8Bit().constData());
            return 3;
        }
        std::printf("%s", buildScript(s, argv[3], "12345").toLocal8Bit().constData());
        return 0;
    }
    QTemporaryDir tmp;
    QDir(tmp.path()).mkpath("data dir/unweighted");
    QDir(tmp.path()).mkpath("data dir/weighted");
    QFile mask(tmp.filePath("data dir/my mask.nii"));
    mask.open(QIODevice::WriteOnly);
    mask.write("x");
    mask.close();
    const QString appDir = tmp.filePath("PAGANI app");

    Settings s;
    s.workingDir = tmp.filePath("data dir");
    s.maskFile = mask.fileName();
    s.maskThreshold = "0.3";
    s.construct = true;
    s.thresholds = "0.3 0.4";
    s.globalLp = true;
    s.nodalEfficiency = true;
    s.globalCp = true;
    s.gamma = s.lambda = s.sigma = true;
    s.modularity = true;
    s.participation = true;
    s.degree = s.betweenness = s.eigenvector = s.nodalCp = true;
    CHECK(validate(s).isEmpty());

    // G-01: correlation threshold passed as "r"
    auto cmds = buildCommands(s, appDir);
    const Command* cm = find(cmds, "CUCorMat");
    CHECK(cm && cm->args.size() == 8 && cm->args[5] == "r" && cm->args[3] == "n");
    s.sparsityThreshold = true;
    s.thresholds = "5 10";
    cmds = buildCommands(s, appDir);
    cm = find(cmds, "CUCorMat");
    CHECK(cm && cm->args[5] == "s");
    // Lp flags and random networks
    const Command* lp = find(cmds, "CUBFW_Lp");
    CHECK(lp && lp->args[1] == "10" && lp->args[2] == "gnl");
    const Command* cp = find(cmds, "Cp");
    CHECK(cp && cp->args.size() == 3 && cp->args[2] == "gnk");
    CHECK(find(cmds, "SmallWorldProperty") && find(cmds, "PC_CPU") && find(cmds, "CUBC") && find(cmds, "ConvertNII"));
    CHECK(cmds.last().program.contains("ConvertNII"));

    // G-02: weighted networks use exefiles_weighted for every tool
    s.weighted = true;
    s.betweenness = false;
    cmds = buildCommands(s, appDir);
    for (const auto& c : cmds) CHECK(c.program.contains("exefiles_weighted"));
    cp = find(cmds, "Cp");
    CHECK(cp && cp->args.size() == 4 && cp->args[2] == "2");
    s.betweenness = true;
    CHECK(!validate(s).isEmpty());  // BC not available for weighted networks
    s.betweenness = false;

    // G-03: nodal efficiency alone still converts to NIfTI
    Settings n1;
    n1.workingDir = s.workingDir;
    n1.maskFile = s.maskFile;
    n1.weighted = true;
    n1.nodalEfficiency = true;
    cmds = buildCommands(n1, appDir);
    CHECK(cmds.size() == 2 && cmds[1].program.contains("ConvertNII"));

    // dependency validation
    Settings bad = n1;
    bad.sigma = true;
    CHECK(!validate(bad).isEmpty());
    bad = n1;
    bad.participation = true;
    CHECK(!validate(bad).isEmpty());
    bad = n1;
    bad.construct = true;
    bad.thresholds = "1.5";
    CHECK(!validate(bad).isEmpty());
    bad.thresholds = "-0.1";
    CHECK(!validate(bad).isEmpty());  // weighted networks need positive weights
    bad.weighted = false;
    CHECK(validate(bad).isEmpty());

    // script round trip (paths with spaces)
    for (int variant = 0; variant < 4; ++variant) {
        Settings t = s;
        t.weighted = variant & 1;
        t.highResolution = variant == 2;
        t.groupNetwork = variant == 1;
        t.fisherAverage = variant == 1;
        t.saveCorMatrix = variant == 3;
        t.modularityAlgorithm = variant % 3;
        t.cpType = variant == 3 ? 1 : 2;
        const QString script = buildScript(t, appDir, "4242");
        CHECK(script.contains("\"" + QDir::toNativeSeparators(t.workingDir) + "\""));
        // one seed for all tools when random networks are used
        CHECK(script.contains("if not defined PAGANI_SEED set \"PAGANI_SEED=4242\"") == usesRandomNetworks(t));
        Settings back;
        QString msg;
        CHECK(parseScript(script, back, &msg));
        CHECK(same(t, back));
    }

    // random-network seed only when random networks are analysed
    {
        Settings t = s;
        CHECK(usesRandomNetworks(t));
        t.gamma = t.lambda = t.sigma = t.modularity = false;
        CHECK(!usesRandomNetworks(t));
        CHECK(!buildScript(t, appDir, "4242").contains("PAGANI_SEED"));
        t.modularity = true;
        t.randomNetworks = "0";
        CHECK(!usesRandomNetworks(t));
        CHECK(newRandomSeed() != newRandomSeed());
    }

    // command parsing without the embedded settings (older scripts)
    {
        Settings t = s;
        t.weighted = false;
        t.betweenness = true;
        t.groupNetwork = true;
        t.fisherAverage = true;
        QString script = buildScript(t, appDir);
        QStringList lines = script.split("\r\n");
        for (int i = lines.size() - 1; i >= 0; --i)
            if (lines[i].startsWith("rem PAGANI-SETTINGS")) lines.removeAt(i);
        Settings back;
        QString msg;
        CHECK(parseScript(lines.join("\r\n"), back, &msg));
        CHECK(back.construct && back.groupNetwork && back.fisherAverage && back.sparsityThreshold);
        CHECK(back.thresholds == "5 10");
        CHECK(back.globalLp && back.nodalEfficiency && back.lambda && back.gamma && back.sigma);
        CHECK(back.modularity && back.participation && back.degree && back.betweenness && back.eigenvector);
        CHECK(QDir::cleanPath(back.workingDir) == QDir::cleanPath(QDir::toNativeSeparators(t.workingDir)) ||
              QDir::fromNativeSeparators(back.workingDir) == QDir::fromNativeSeparators(t.workingDir));
        CHECK(back.maskFile == QDir::toNativeSeparators(t.maskFile));
    }

    // PAGANI 1.x / 2.0 scripts shipped with the old package
    if (argc > 1) {
        for (const char* name : {"script.bat", "scriptweighted.bat", "script_test.bat"}) {
            QFile f(QDir(argv[1]).filePath(name));
            if (!f.open(QIODevice::ReadOnly)) continue;
            Settings back;
            QString msg;
            CHECK(parseScript(QString::fromLocal8Bit(f.readAll()), back, &msg));
            if (QString(name) == "script.bat") {
                CHECK(!back.weighted && back.construct && back.groupNetwork && !back.fisherAverage);
                CHECK(back.workingDir == "E:/revised/data/4mm" || back.workingDir == "E:\\revised\\data\\4mm");
                CHECK(back.globalLp && back.lambda && back.gamma && back.sigma && back.modularity && back.betweenness);
            }
            if (QString(name) == "scriptweighted.bat") {
                CHECK(back.weighted && back.construct && back.sparsityThreshold);
                CHECK(back.globalCp && back.nodalCp && back.cpType == 2 && back.degree && back.eigenvector);
            }
        }
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all GUI pipeline checks passed\n");
    return 0;
}
