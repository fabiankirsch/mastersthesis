---
documentclass: article
smart: yes
fontsize: 11pt
geometry: [twoside,a4paper,margin=4.0cm]
indent: true
figPrefix:
  - "Figure"
  - "Figures"
secPrefix:
  - "Section"
  - "Sections"
tblPrefix:
  - "Table"
  - "Tables"
lstPrefix:
  - "Listing"
  - "Listings"
header-includes:
  - \usepackage{fvextra}
  - \usepackage{setspace}
  - \doublespacing
  - \usepackage{fancyhdr}
  - \usepackage[hang, flushmargin]{footmisc}
  - \RecustomVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}
  - \pagestyle{fancy}
  - \fancyhead[CE,CO]{\scshape \leftmark}
  - \fancyhead[LE,RO]{}
  - \fancyhead[LO,RE]{}
  - \usepackage{pdfpages}
  - \usepackage{indentfirst}
  - \usepackage{hyperref}
---

\begin{titlepage}
    \begin{center}
        \vspace*{1cm}

        \huge Requirements, preparations and implementation of an exemplary machine learning pipeline for HCI
        \vspace{1.0cm}

        \normalsize Master's thesis by\\
        \Large Fabian Kirsch\\

        \vspace{1.0cm}

	   \vfill\normalsize First Supervisor: \\
       Prof. Dr.-Ing. Matthias Rötting\\
	   \vfill\normalsize Second Supervisor: \\
       Sarah-Christin Freytag, M.Sc.\\


        \vspace{1.0cm}
        \includegraphics[width=0.3\textwidth]{figures/TU-Berlin-Logo.png}


        \normalsize
        Human Factors \\
        Faculty V\\
        Technical University of Berlin\\
		16.09.2019\\

    \end{center}
\end{titlepage}


\pagestyle{empty}


\pagenumbering{gobble}
\setcounter{page}{0}

\cleardoublepage

\begin{center}
\noindent\begin{minipage}[c][\textheight][c]{0.75\textwidth}
    Hiermit erkläre ich, dass ich die vorliegende Arbeit selbstständig und eigenhändig sowie ohne unerlaubte fremde Hilfe und ausschließlich unter Verwendung der aufgeführten Quellen und Hilfsmittel angefertigt habe.

    \vspace{1cm}
    Berlin, den \hrulefill

    \vspace{1cm}
    \hrulefill \\
    Unterschrift
\end{minipage}
\end{center}

\cleardoublepage

# Abstract {.unnumbered}

The goal of this master's thesis is to explore machine learning methods suitable for enhancing human computer interaction (HCI). Fundamental concepts of machine learning are introduced and the general composition of a machine learning pipeline is explained. An entire machine learning pipeline was implemented focusing on methods that require little data-specific adaption and potentially generalize well to a wide range of use cases in the HCI domain. For the modeling layer an LSTM (a recurrent neural network) was chosen and different architectures and configurations were tested and compared. A public human activity recognition (HAR) time series data set was used for testing the pipeline. To enhance understanding for the reader the machine learning pipeline was integrated into the digital version of this thesis and an entirely reproducible script is provided. The best performing model has an accuracy of ~90% on the HAR data set, which already provides strong evidence for the benefits of generalizable machine learning methods in HCI. Specific suggestions for further pipeline enhancements are presented. A key learning from this thesis is that the field of machine learning is vast and understanding even one type of algorithm well requires significant time. Much time was spent on manually finding a suitable architecture and configuration for the LSTM. Therefore, some frameworks that automate this process are recommended given that sufficient computational resources are available.


\cleardoublepage


# Zusammenfassung {.unnumbered}

Das Ziel dieser Masterarbeit ist es Methoden des Machine-Learnings zu untersuchen, die sich zur Verbesserung von Human-Computer-Interfaces (HCI) eignen. Grundlegende Konzepte des Machine-Learnings werden vorgestellt und der Grundaufbau einer Machine-Learning-Pipeline wird erklärt. Zudem wurde eine vollständige Machine-Learning-Pipeline implementiert mit dem Fokus auf Methoden, die wenig manuelle Anpassung für den spezifischen Use Case benötigen und dadurch leicht auf andere Use Cases übertragbar sind. Ein LSTM (eine Form des Recurrent Neural Network) wurde implementiert und verschiedene Architekturen und Konfigurationen wurden getestet und verglichen. Ein öffentlich verfügbarer Human-Activity-Recognition (HAR) Datensatz wurde zum Testen der Pipeline verwendet. Um dem Leser das Verstehen einer Machine-Learning-Pipeline zu erleichtern wurden Text und Software Code in der digitalen Version der Masterarbeit zu einem komplett reproduzierbaren Skript integriert. Die Accuracy des am besten funktionierenden Models auf dem HAR Datensatz ist um die 90%. Dies zeigt eindeutig das Potential von generalisierbaren Machine-Learning Methoden zur Verbesserung von HCI. Zudem werden konkrete Verbesserungsvorschläge für die Pipeline vorgestellt. Eine zentrale Lernerfahrung aus dieser Arbeit ist, dass das Feld des Machine-Learnings gewaltig groß ist und auch nur eine Art von Algorithmus zu verstehen schon sehr zeitaufwändig ist. Viel Zeit wurde investiert um manuell eine gute Architektur und Konfiguration für das LSTM zu finden. Daher werden für den Fall, dass ausreichend Rechenkapazitäten vorhanden sind einige Frameworks empfohlen, die diesen Vorgang automatisieren.

\cleardoublepage

\tableofcontents

\cleardoublepage

\listoffigures

\cleardoublepage

\listoftables


\cleardoublepage

\pagestyle{fancy}
\setcounter{page}{1}
\pagenumbering{arabic}
