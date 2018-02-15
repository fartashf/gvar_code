BIBTEX = bibtex
#BIBTEX = echo
LATEX = latex
#PS2PDF = ps2pdf
PDFLATEX = pdflatex -output-format=pdf --shell-escape -interaction=nonstopmode -halt-on-error

# DVIPDF = dvipdf
# DVIPS = dvips
# DVIFLAGS = -Ppdf -t letter -G0
#DVIFLAGS = -t letter

SRCNAME = nips_2017

TEXSOURCES = $(wildcard *.tex) $(wildcard figures/*.tex)
BIBSOURCES = $(wildcard *.bib)
SOURCES = $(TEXSOURCES) $(BIBSOURCES)

#OUTPUTS = $(SRCNAME).dvi $(SRCNAME).ps $(SRCNAME).pdf
OUTPUTS = $(SRCNAME).pdf
BYPRODUCTS = $(SRCNAME).aux $(SRCNAME).bbl $(SRCNAME).blg $(SRCNAME).brf

ARXIV_TAR = $(SRCNAME).tar.gz

all: $(OUTPUTS)

#pdf $(SRCNAME).pdf: $(SRCNAME).ps
#	$(PS2PDF) $(SRCNAME).ps
#pdf $(SRCNAME).pdf: $(SRCNAME).dvi
#	$(DVIPDF) $(DVIFLAGS) $(SRCNAME).dvi $(SRCNAME).pdf

pdf $(SRCNAME).pdf: $(SOURCES)
	$(PDFLATEX) $(SRCNAME)
	# $(BIBTEX) $(SRCNAME)
	$(PDFLATEX) $(SRCNAME)
	$(PDFLATEX) $(SRCNAME)

# ps $(SRCNAME).ps: $(SRCNAME).dvi
# 	$(DVIPS) $(DVIFLAGS) $(SRCNAME).dvi -o $(SRCNAME).ps

# #dvi $(SRCNAME).dvi: $(SRCNAME).aux $(SRCNAME).bbl $(SOURCES)
# dvi $(SRCNAME).dvi: $(SOURCES)
# 	$(LATEX) $(SRCNAME)
# 	$(BIBTEX) $(SRCNAME)
# 	$(LATEX) $(SRCNAME)
# 	$(LATEX) $(SRCNAME)

# $(SRCNAME).bbl: $(TEXSOURCES) $(BIBSOURCES)
# 		$(LATEX) $(SRCNAME)
# 		$(BIBTEX) $(SRCNAME)

clean:
	rm -f $(OUTPUTS) $(BYPRODUCTS)

arxiv:
	tar -cvzf $(ARXIV_TAR) $(TEXSOURCES) $(wildcard images/*) $(wildcard *.sty) $(SRCNAME).bbl
