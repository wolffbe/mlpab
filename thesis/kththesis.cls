%% forked from https://gits-15.sys.kth.se/giampi/kthlatex kthlatex-0.2rc4 on 2020-02-13
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{kththesis}[2025/05/25 KTH Thesis formatting]

\newtoggle{digitaloutput}
\DeclareOption{digitaloutput}{\toggletrue{digitaloutput}}

% Add optional nomenclature pages following glossary
\newif\ifnomenclature
\DeclareOption{nomenclature}{\nomenclaturetrue}
\nomenclaturefalse

\newif\ifinswedish
\DeclareOption{english}{}
\DeclareOption{swedish}{\inswedishtrue}

\newtoggle{gfivepaper}
\DeclareOption{a4paper}{}
\DeclareOption{g5paper}{\toggletrue{gfivepaper}}

% options for bachelor or master thesis - not currently used
\newtoggle{bachelor}
\DeclareOption{bachelor}{\toggletrue{bachelor}}
\DeclareOption{master}{\togglefalse{bachelor}}


\newtoggle{biblatex}
\DeclareOption{bibtex}{\togglefalse{biblatex}}
\DeclareOption{biblatex}{\toggletrue{biblatex}}

\newtoggle{includeExtraAbstracts}

\DeclareOption{includeExtraAbstracts}{\toggletrue{includeExtraAbstracts}}

%% Send any unknown option to the report class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}

\ProcessOptions\relax

% Some macros to make access to the options easier
\let\ifCLASSOPTIONincludeExtraAbstracts\ifkth@includeExtraAbstracts

\iftoggle{gfivepaper}{
  \LoadClass[10pt]{report}
}{
  \LoadClass[12pt]{report}
}

% Load all external packages needed by kththesis.cls
%\RequirePackage{./kth/kth-packages}
\input{./kth/kth-packages}

\ifxeorlua
  \RequirePackage{polyglossia}
  \ifinswedish
    \setmainlanguage{swedish}
    \setotherlanguage[variant=us]{english}
    \setotherlanguage[variant=french, frenchpart=false]{french}
    \setotherlanguage[variant=spanish]{spanish}
    %\setotherlanguages{italian,norsk,german,danish,dutch,estonian}
        \setotherlanguage[variant=monotonic]{greek}
    \setotherlanguage[numerals=western]{hindi}
    \setotherlanguage[numerals=arabic, forceheadingpunctuation=false]{russian}
    \setotherlanguage[variant=simplified]{chinese} % numerals=arabic

    \setotherlanguages{italian,norsk,german,danish,dutch,estonian, finnish}

  \else
    \setmainlanguage[variant=us]{english}
    \setotherlanguage[variant=french, frenchpart=false]{french}
    \setotherlanguage[variant=spanish]{spanish}
    \setotherlanguage[variant=bokmal]{norwegian}
    \setotherlanguage[variant=german, spelling=new]{german}
    \setotherlanguage[variant=monotonic]{greek}
    \setotherlanguage[numerals=western]{hindi}
    \setotherlanguage[numerals=arabic, forceheadingpunctuation=false]{russian}
    \setotherlanguage[variant=simplified]{chinese} % numerals=arabic
    
    %\setotherlanguages{swedish,italian,danish,dutch,estonian}
    \setotherlanguages{swedish,italian,danish,dutch,estonian, finnish}
    %\setotherlanguage{swedish}
  \fi
\else
  \RequirePackage[utf8]{inputenc}
  \ifinswedish
    % can replace USenglish with UKenglish if desired
    \RequirePackage[main=swedish,english,french,spanish,italian,norsk,ngerman,danish,dutch,estonian]{babel}
  \else
    % can replace USenglish with UKenglish if desired
    \RequirePackage[main=english,swedish,french,spanish,italian,norsk,ngerman,danish,dutch,estonian]{babel}
  \fi
\fi

\iftoggle{gfivepaper}{
\RequirePackage[paperwidth=169mm,paperheight=239mm,twoside,
  top=30mm,bottom=30mm,inner=36mm,outer=18mm,
  headsep=10mm,headheight=5mm]{geometry} % Set paper and contents dimensions
}{
\RequirePackage[a4paper,twoside,
  top=37mm,bottom=50mm,inner=45mm,outer=35mm,
  headsep=15mm,headheight=6mm,footskip=25mm]{geometry} % Set paper and contents dimensions
}


% Set up fonts
\input{./kth/kth-fonts}

% Define page layout and colors
%\RequirePackage{./kth/kth-layout}
\input{./kth/kth-layout}

% 5. Define all custom commands and environments
%\RequirePackage{./kth/kth-commands}
\input{./kth/kth-commands}

% get all of the information about the school codes and program codes
\input{lib/schools_and_programs.ins}

\newcommand{\MONTH}{%
  \ifinswedish
  \ifcase\the\month
  \or januari% 1
  \or februari% 2
  \or mars% 3
  \or april% 4
  \or maj% 5
  \or juni% 6
  \or juli% 7
  \or augusti% 8
  \or september% 9
  \or oktober% 10
  \or november% 11
  \or december% 12
  \fi
  \else
  \ifcase\the\month
  \or January% 1
  \or February% 2
  \or March% 3
  \or April% 4
  \or May% 5
  \or June% 6
  \or July% 7
  \or August% 8
  \or September% 9
  \or October% 10
  \or November% 11
  \or December% 12
  \fi
\fi
}



% Input the sanity checks to be done at the end of the document
\input{kth/kth-sanity-checks}
\endinput
