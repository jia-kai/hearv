# $File: Makefile
# $Date: Sat Nov 29 10:43:43 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

TARGET = report
PDFDEPS = build $(addprefix build/,$(wildcard *.tex)) \
		  $(addprefix build/,$(wildcard *.bib)) build/res

all: view

build/$(TARGET).pdf: $(PDFDEPS)
	cd build && latexmk -xelatex -8bit -shell-escape $(TARGET).tex

build/%: %
	[ -h $@ ] || ln -s ../$< build/

build:
	mkdir $@

view: build/$(TARGET).pdf
	okular build/$(TARGET).pdf

rebuild: clean build/$(TARGET).pdf

clean:
	rm -rf build

.PHONY: all view clean rebuild

# vim: ft=make

