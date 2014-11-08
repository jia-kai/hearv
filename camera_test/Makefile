# $File: Makefile
# $Date: Sat Nov 08 11:29:55 2014 +0800
# $Author: jiakai <jia.kai66@gmail.com>

TARGETS := $(shell cat .gitignore)

CXX := g++ -std=c++11

PKGCONFIG_LIBS := opencv glu
override OPTFLAG ?= -O2

override CXXFLAGS += \
	-ggdb \
	-Wall -Wextra -Wnon-virtual-dtor -Wno-unused-parameter -Winvalid-pch \
	-Wno-unused-local-typedefs \
	$(CPPFLAGS) $(OPTFLAG) \
	$(shell pkg-config --cflags $(PKGCONFIG_LIBS)) 
override LDFLAGS += $(shell pkg-config --libs $(PKGCONFIG_LIBS)) -lglut
override V ?= @

all: $(TARGETS)

%: %.cpp
	@echo "[compile] $@"
	$(V)$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -rf $(TARGETS)

debug: 
	OPTFLAG=-O0 make -j4


.PHONY: all clean debug

# vim: ft=make

