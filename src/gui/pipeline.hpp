// PAGANI GUI - analysis settings, command generation and script I/O.
// Independent of the widgets so it can be unit tested.
#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

namespace pagani_gui {

struct Settings {
    // inputs
    QString workingDir;     // folder with the 4D images (and the network sub-folders)
    QString maskFile;
    QString maskThreshold = "0.2";
    bool weighted = false;

    // network construction
    bool construct = false;
    bool highResolution = false;   // CUCorMatHr (PAGANI 2.0 option)
    bool sparsityThreshold = false;  // false: correlation threshold
    QString thresholds;            // one or more values separated by spaces
    bool groupNetwork = false;     // standard construction only
    bool fisherAverage = false;
    bool saveCorMatrix = false;    // standard construction only

    // global metrics
    bool globalLp = false;
    bool globalCp = false;
    bool gamma = false;
    bool lambda = false;
    bool sigma = false;
    bool modularity = false;
    int modularityAlgorithm = 0;   // 0 Louvain, 1 Newman (GPU), 2 Newman (CPU)
    int cpType = 2;                // weighted Cp: 1 Barrat, 2 Onnela
    QString randomNetworks = "10";

    // nodal metrics
    bool nodalEfficiency = false;
    bool betweenness = false;      // unweighted networks only
    bool nodalCp = false;
    bool degree = false;
    bool eigenvector = false;
    bool participation = false;

    QString toJson() const;
    static bool fromJson(const QString& json, Settings& out);
};

struct Command {
    QString title;      // short description for the log
    QString program;    // absolute path of the executable
    QStringList args;
};

// Directory that contains exefiles\ and exefiles_weighted\ (the GUI folder).
QString toolDirectory(const QString& appDir, bool weighted);
QString toolPath(const QString& appDir, bool weighted, const QString& name);
// Folder holding the .csr networks for the current settings.
QString networkDirectory(const Settings& s);

// Validate settings; returns an empty list when everything is fine.
QStringList validate(const Settings& s);

// Commands to run, in order.
QVector<Command> buildCommands(const Settings& s, const QString& appDir);

// True when the commands analyse random networks (lambda, gamma, sigma or the
// modularity of random networks).
bool usesRandomNetworks(const Settings& s);

// A new seed for the random networks of one run (decimal, for PAGANI_SEED).
// All tools of a run use the same seed, so lambda, gamma, sigma and the Q of
// random networks are computed from the same random networks.
QString newRandomSeed();

// Windows batch script with the commands (and the settings embedded as a
// comment so that the GUI can restore them exactly).  With a non-empty
// `seed`, the script sets PAGANI_SEED to it (unless it is already set), so
// re-running the script reproduces the random networks.
QString buildScript(const Settings& s, const QString& appDir, const QString& seed = QString());

// Restore settings from a script written by PAGANI 2.1, or reconstruct them
// from the commands of a PAGANI 1.x/2.0 script.  Returns false and a message
// when the script cannot be interpreted.
bool parseScript(const QString& text, Settings& out, QString* message);

// Split a command line honouring double quotes.
QStringList splitCommandLine(const QString& line);

}  // namespace pagani_gui
