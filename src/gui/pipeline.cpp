#include "pipeline.hpp"

#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <cstring>

namespace pagani_gui {

namespace {

const char* kSettingsTag = "rem PAGANI-SETTINGS ";

QString exeName(const QString& base) {
#ifdef Q_OS_WIN
    return base + ".exe";
#else
    return base;
#endif
}

QStringList thresholdValues(const QString& s) {
    return s.split(QRegularExpression("[\\s,;]+"), Qt::SkipEmptyParts);
}

bool isNumber(const QString& s, double* v = nullptr) {
    bool ok = false;
    double x = s.toDouble(&ok);
    if (v) *v = x;
    return ok;
}

QString quoteArg(const QString& a) {
    if (a.isEmpty()) return "\"\"";
    if (a.contains(' ') || a.contains('\t') || a.contains('&') || a.contains('(') || a.contains(')') ||
        a.contains('^') || a.contains('%'))
        return '"' + a + '"';
    return a;
}

}  // namespace

// --------------------------------------------------------------------- JSON
QString Settings::toJson() const {
    QJsonObject o;
    o["version"] = "2.1";
    o["workingDir"] = workingDir;
    o["maskFile"] = maskFile;
    o["maskThreshold"] = maskThreshold;
    o["weighted"] = weighted;
    o["construct"] = construct;
    o["highResolution"] = highResolution;
    o["sparsityThreshold"] = sparsityThreshold;
    o["thresholds"] = thresholds;
    o["groupNetwork"] = groupNetwork;
    o["fisherAverage"] = fisherAverage;
    o["saveCorMatrix"] = saveCorMatrix;
    o["globalLp"] = globalLp;
    o["globalCp"] = globalCp;
    o["gamma"] = gamma;
    o["lambda"] = lambda;
    o["sigma"] = sigma;
    o["modularity"] = modularity;
    o["modularityAlgorithm"] = modularityAlgorithm;
    o["cpType"] = cpType;
    o["randomNetworks"] = randomNetworks;
    o["nodalEfficiency"] = nodalEfficiency;
    o["betweenness"] = betweenness;
    o["nodalCp"] = nodalCp;
    o["degree"] = degree;
    o["eigenvector"] = eigenvector;
    o["participation"] = participation;
    return QString::fromUtf8(QJsonDocument(o).toJson(QJsonDocument::Compact));
}

bool Settings::fromJson(const QString& json, Settings& s) {
    QJsonParseError err;
    QJsonDocument d = QJsonDocument::fromJson(json.toUtf8(), &err);
    if (err.error != QJsonParseError::NoError || !d.isObject()) return false;
    QJsonObject o = d.object();
    Settings t;
    t.workingDir = o["workingDir"].toString();
    t.maskFile = o["maskFile"].toString();
    t.maskThreshold = o["maskThreshold"].toString(t.maskThreshold);
    t.weighted = o["weighted"].toBool();
    t.construct = o["construct"].toBool();
    t.highResolution = o["highResolution"].toBool();
    t.sparsityThreshold = o["sparsityThreshold"].toBool();
    t.thresholds = o["thresholds"].toString();
    t.groupNetwork = o["groupNetwork"].toBool();
    t.fisherAverage = o["fisherAverage"].toBool();
    t.saveCorMatrix = o["saveCorMatrix"].toBool();
    t.globalLp = o["globalLp"].toBool();
    t.globalCp = o["globalCp"].toBool();
    t.gamma = o["gamma"].toBool();
    t.lambda = o["lambda"].toBool();
    t.sigma = o["sigma"].toBool();
    t.modularity = o["modularity"].toBool();
    t.modularityAlgorithm = o["modularityAlgorithm"].toInt();
    t.cpType = o["cpType"].toInt(2);
    t.randomNetworks = o["randomNetworks"].toString(t.randomNetworks);
    t.nodalEfficiency = o["nodalEfficiency"].toBool();
    t.betweenness = o["betweenness"].toBool();
    t.nodalCp = o["nodalCp"].toBool();
    t.degree = o["degree"].toBool();
    t.eigenvector = o["eigenvector"].toBool();
    t.participation = o["participation"].toBool();
    s = t;
    return true;
}

// ------------------------------------------------------------------- paths
QString toolDirectory(const QString& appDir, bool weighted) {
    return QDir(appDir).filePath(weighted ? "exefiles_weighted" : "exefiles");
}

QString toolPath(const QString& appDir, bool weighted, const QString& name) {
    return QDir::toNativeSeparators(QDir(toolDirectory(appDir, weighted)).filePath(exeName(name)));
}

QString networkDirectory(const Settings& s) {
    return QDir::toNativeSeparators(QDir(s.workingDir).filePath(s.weighted ? "weighted" : "unweighted"));
}

// -------------------------------------------------------------- validation
static bool anyNodal(const Settings& s) {
    return s.nodalEfficiency || s.betweenness || s.nodalCp || s.degree || s.eigenvector || s.participation ||
           s.modularity;
}

static bool needsRandom(const Settings& s) { return s.gamma || s.lambda || s.sigma; }

QStringList validate(const Settings& s) {
    QStringList e;
    if (s.workingDir.trimmed().isEmpty()) e << "Working directory is empty.";
    else if (!QFileInfo(s.workingDir).isDir()) e << "Working directory does not exist: " + s.workingDir;
    const bool needMask = s.construct || anyNodal(s);
    if (needMask) {
        if (s.maskFile.trimmed().isEmpty()) e << "Mask file is empty.";
        else if (!QFileInfo(s.maskFile).isFile()) e << "Mask file does not exist: " + s.maskFile;
        double t;
        if (!isNumber(s.maskThreshold, &t)) e << "Mask threshold must be a number.";
    }
    if (s.construct) {
        const QStringList v = thresholdValues(s.thresholds);
        if (v.isEmpty()) e << "Enter at least one threshold value.";
        for (const QString& x : v) {
            double t;
            if (!isNumber(x, &t)) { e << "Invalid threshold value: " + x; continue; }
            if (!s.sparsityThreshold && !(t > -1.0 && t < 1.0)) e << "Correlation thresholds must be between -1 and 1: " + x;
            else if (!s.sparsityThreshold && s.weighted && t < 0.0)
                e << "Weighted networks need correlation thresholds >= 0 (the edge weights are r and must be positive): " + x;
            if (s.sparsityThreshold && !(t > 0.0 && t <= 100.0)) e << "Sparsity thresholds are percentages in (0, 100]: " + x;
        }
    } else if (!s.workingDir.isEmpty() && QFileInfo(s.workingDir).isDir() && !QFileInfo(networkDirectory(s)).isDir() &&
               (s.globalLp || s.globalCp || needsRandom(s) || anyNodal(s))) {
        e << "Network folder " + networkDirectory(s) + " does not exist. Enable \"Network Construction\" first.";
    }
    if (needsRandom(s)) {
        bool ok = false;
        int n = s.randomNetworks.toInt(&ok);
        if (!ok || n < 1) e << "Gamma, lambda and small-world sigma need at least 1 random network.";
    } else if (s.modularity && !s.randomNetworks.trimmed().isEmpty()) {
        bool ok = false;
        int n = s.randomNetworks.toInt(&ok);
        if (!ok || n < 0) e << "Number of random networks must be a non-negative integer.";
    }
    if (s.sigma && !(s.gamma && s.lambda)) e << "Small-world sigma requires gamma and lambda.";
    if (s.participation && !s.modularity) e << "Participation coefficient requires Modularity.";
    if (s.betweenness && s.weighted) e << "Betweenness centrality is only available for unweighted networks.";
    if (!s.construct && !s.globalLp && !s.globalCp && !needsRandom(s) && !anyNodal(s))
        e << "Nothing to do: select at least one step.";
    return e;
}

// ---------------------------------------------------------------- commands
QVector<Command> buildCommands(const Settings& s, const QString& appDir) {
    QVector<Command> cmds;
    const bool w = s.weighted;
    const QString net = networkDirectory(s);
    const QString work = QDir::toNativeSeparators(s.workingDir);
    const QString mask = QDir::toNativeSeparators(s.maskFile);
    const QString nrand = s.randomNetworks.trimmed().isEmpty() ? "0" : s.randomNetworks.trimmed();

    if (s.construct) {
        Command c;
        const QString type = s.sparsityThreshold ? "s" : "r";
        if (s.highResolution) {
            c.title = "Network construction (high resolution)";
            c.program = toolPath(appDir, w, "CUCorMatHr");
            c.args << work << mask << s.maskThreshold << "n" << type << "p";
        } else {
            c.title = "Network construction";
            c.program = toolPath(appDir, w, "CUCorMat");
            const QString avg = s.groupNetwork ? (s.fisherAverage ? "bf" : "bn") : "n";
            c.args << work << mask << s.maskThreshold << avg << (s.saveCorMatrix ? "y" : "n") << type;
        }
        c.args << thresholdValues(s.thresholds);
        cmds << c;
    }
    if (s.globalLp || s.nodalEfficiency || s.lambda || s.sigma) {
        QString f;
        if (s.globalLp) f += 'g';
        if (s.nodalEfficiency) f += 'n';
        if (s.lambda || s.sigma) f += 'l';
        cmds << Command{"Characteristic path length / nodal efficiency", toolPath(appDir, w, "CUBFW_Lp"),
                        {net, (s.lambda || s.sigma) ? nrand : "0", f}};
    }
    if (s.globalCp || s.nodalCp || s.gamma || s.sigma) {
        QString f;
        if (s.globalCp) f += 'g';
        if (s.nodalCp) f += 'n';
        if (s.gamma || s.sigma) f += 'k';
        Command c{"Clustering coefficient", toolPath(appDir, w, "Cp"), {net, (s.gamma || s.sigma) ? nrand : "0"}};
        if (w) c.args << QString::number(s.cpType == 1 ? 1 : 2);
        c.args << f;
        cmds << c;
    }
    if (s.sigma) cmds << Command{"Small-world property", toolPath(appDir, w, "SmallWorldProperty"), {net}};
    if (s.modularity) {
        QString exe = "Louvain_Modularity";
        if (s.modularityAlgorithm == 1) exe = w ? "CUModularity" : "CUmodularity";
        else if (s.modularityAlgorithm == 2) exe = "Newman_Modularity_CPU";
        cmds << Command{"Modularity", toolPath(appDir, w, exe), {net, nrand}};
    }
    if (s.participation) cmds << Command{"Participation coefficient", toolPath(appDir, w, "PC_CPU"), {net}};
    if (s.degree) cmds << Command{"Degree centrality", toolPath(appDir, w, "Degree"), {net}};
    if (s.betweenness && !w) cmds << Command{"Betweenness centrality", toolPath(appDir, false, "CUBC"), {net}};
    if (s.eigenvector) cmds << Command{"Eigenvector centrality", toolPath(appDir, w, "CUEC"), {net}};
    if (anyNodal(s))
        cmds << Command{"Convert nodal metrics to NIfTI", toolPath(appDir, w, "ConvertNII"), {net, mask, s.maskThreshold}};
    return cmds;
}

QString buildScript(const Settings& s, const QString& appDir) {
    QString out;
    out += "@echo off\r\n";
    out += "rem PAGANI Toolkit 2.1 analysis script (generated by the PAGANI GUI)\r\n";
    out += kSettingsTag + s.toJson() + "\r\n";
    out += s.weighted ? "echo weightednetworks\r\n" : "echo unweightednetworks\r\n";
    for (const Command& c : buildCommands(s, appDir)) {
        out += "\r\necho ==== " + c.title + "\r\n";
        QStringList parts;
        parts << quoteArg(c.program);
        for (const QString& a : c.args) parts << quoteArg(a);
        out += parts.join(' ') + "\r\n";
        out += "if errorlevel 1 goto failed\r\n";
    }
    out += "\r\necho PAGANI: all steps finished.\r\ngoto :eof\r\n";
    out += ":failed\r\necho PAGANI: a step failed, see the messages above.\r\nexit /b 1\r\n";
    return out;
}

// ----------------------------------------------------------------- parsing
QStringList splitCommandLine(const QString& line) {
    QStringList out;
    QString cur;
    bool inq = false, have = false;
    for (QChar ch : line) {
        if (ch == '"') { inq = !inq; have = true; continue; }
        if (!inq && ch.isSpace()) {
            if (have) { out << cur; cur.clear(); have = false; }
            continue;
        }
        cur += ch;
        have = true;
    }
    if (have) out << cur;
    return out;
}

static QString stripNetworkFolder(const QString& dir) {
    QString d = dir;
    d.replace('\\', '/');
    while (d.endsWith('/')) d.chop(1);
    for (const char* sub : {"/unweighted", "/weighted", "/weightedKendall"})
        if (d.endsWith(sub, Qt::CaseInsensitive)) return QDir::toNativeSeparators(d.left(d.size() - static_cast<int>(strlen(sub))));
    return QDir::toNativeSeparators(d);
}

bool parseScript(const QString& text, Settings& out, QString* message) {
    const QStringList lines = text.split(QRegularExpression("\\r?\\n"));
    for (const QString& l : lines) {
        if (l.startsWith(kSettingsTag)) {
            if (Settings::fromJson(l.mid(static_cast<int>(strlen(kSettingsTag))), out)) {
                if (message) *message = "Settings restored from a PAGANI 2.1 script.";
                return true;
            }
        }
    }
    // PAGANI 1.x / 2.0 script: reconstruct from the commands
    Settings s;
    s.maskThreshold.clear();
    s.randomNetworks.clear();
    int marker = -1;  // 0 unweighted, 1 weighted
    QStringList unknown;
    int recognised = 0;
    for (const QString& raw : lines) {
        const QString line = raw.trimmed();
        if (line.isEmpty() || line.startsWith("rem", Qt::CaseInsensitive) || line.startsWith("@") ||
            line.startsWith("::") || line.startsWith("if errorlevel") || line.startsWith("goto") ||
            line.startsWith(":") || line.startsWith("exit") || line.startsWith("pause", Qt::CaseInsensitive))
            continue;
        QStringList t = splitCommandLine(line);
        if (t.isEmpty()) continue;
        if (t[0].compare("echo", Qt::CaseInsensitive) == 0) {
            if (t.size() > 1 && t[1] == "unweightednetworks") marker = 0;
            if (t.size() > 1 && t[1] == "weightednetworks") marker = 1;
            continue;
        }
        QString prog = t[0];
        prog.replace('\\', '/');
        const bool fromWeightedDir = prog.contains("exefiles_weighted", Qt::CaseInsensitive);
        QString name = QFileInfo(prog).fileName().toLower();
        if (name.endsWith(".exe")) name.chop(4);
        const QStringList a = t.mid(1);
        auto flags = [&](int idx) { return idx < a.size() ? a[idx].toLower() : QString(); };
        if (marker < 0 && fromWeightedDir) marker = 1;
        ++recognised;
        if ((name == "cucormat") && a.size() >= 7) {
            s.construct = true;
            s.highResolution = false;
            s.workingDir = a[0];
            s.maskFile = a[1];
            s.maskThreshold = a[2];
            const QString avg = a[3].toLower();
            s.groupNetwork = avg.startsWith('y') || avg.startsWith('b');
            s.fisherAverage = avg.size() > 1 && avg[1] == 'f';
            s.saveCorMatrix = a[4].toLower() == "y";
            s.sparsityThreshold = a[5].toLower() == "s";
            s.thresholds = a.mid(6).join(' ');
        } else if (name == "cucormathr" && a.size() >= 7) {
            s.construct = true;
            s.highResolution = true;
            s.workingDir = a[0];
            s.maskFile = a[1];
            s.maskThreshold = a[2];
            s.sparsityThreshold = a[4].toLower() == "s";
            s.thresholds = a.mid(6).join(' ');
        } else if ((name == "cubfw_lp" || name == "bfs_mulcpu") && a.size() >= 3) {
            if (s.workingDir.isEmpty()) s.workingDir = stripNetworkFolder(a[0]);
            QString f = flags(2);
            if (f == "b") f = "gn";
            s.globalLp |= f.contains('g');
            s.nodalEfficiency |= f.contains('n');
            s.lambda |= f.contains('l');
            if (a[1] != "0") s.randomNetworks = a[1];
        } else if (name == "cp" && a.size() >= 3) {
            if (s.workingDir.isEmpty()) s.workingDir = stripNetworkFolder(a[0]);
            QString f = a.size() >= 4 ? flags(3) : flags(2);
            if (a.size() >= 4) s.cpType = a[2] == "1" ? 1 : 2;
            if (f == "b") f = "gn";
            s.globalCp |= f.contains('g');
            s.nodalCp |= f.contains('n');
            s.gamma |= f.contains('k');
            if (a[1] != "0") s.randomNetworks = a[1];
        } else if (name == "smallworldproperty") {
            s.sigma = true;
        } else if ((name == "louvain_modularity" || name == "cumodularity" || name == "newman_modularity_gpu" ||
                    name == "newman_modularity_cpu") && !a.isEmpty()) {
            if (s.workingDir.isEmpty()) s.workingDir = stripNetworkFolder(a[0]);
            s.modularity = true;
            s.modularityAlgorithm = name == "louvain_modularity" ? 0 : (name == "newman_modularity_cpu" ? 2 : 1);
            if (a.size() > 1 && a[1] != "0" && s.randomNetworks.isEmpty()) s.randomNetworks = a[1];
        } else if (name == "pc_cpu") {
            s.participation = true;
        } else if (name == "degree") {
            s.degree = true;
        } else if (name == "cubc") {
            s.betweenness = true;
        } else if (name == "cuec") {
            s.eigenvector = true;
        } else if (name == "convertnii" && a.size() >= 2) {
            if (s.workingDir.isEmpty()) s.workingDir = stripNetworkFolder(a[0]);
            s.maskFile = a[1];
            if (a.size() >= 3) s.maskThreshold = a[2];
        } else {
            --recognised;
            unknown << line;
        }
    }
    if (marker < 0 && recognised == 0) {
        if (message) *message = "This file does not look like a PAGANI script.";
        return false;
    }
    s.weighted = marker == 1;
    if (s.maskThreshold.isEmpty()) s.maskThreshold = "0.2";
    if (s.randomNetworks.isEmpty()) s.randomNetworks = "10";
    if (s.betweenness && s.weighted) s.betweenness = false;
    out = s;
    if (message) {
        *message = "Settings reconstructed from an older PAGANI script.";
        if (!unknown.isEmpty()) *message += "\nIgnored lines:\n" + unknown.join('\n');
    }
    return true;
}

}  // namespace pagani_gui
