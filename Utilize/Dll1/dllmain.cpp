/*
 * Simple example c program to write a
 * binary datafile for tecplot.  This example
 * does the following:
 *
 *   1.  Open a datafile called "*.plt"
 *   2.  Assign values for x,y,z,and others.
 *   3.  Write out a hexahedral (brick) zone.
 *   4.  Close the datafile.
 */

 // Internal testing flags
 // RUNFLAGS:none
 // RUNFLAGS:--szl
# define DLLEXPORT extern "C" __declspec(dllexport)
#include "TECIO.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#ifndef NULL
#define NULL 0
#endif

enum fileType_e { FULL = 0, GRID = 1, SOLUTION = 2 };

DLLEXPORT void main(char* Title, char* VariableNames, char* FileName, char* ScratchDir, int Nnum, int Enum,int valnum,
    int* connect,double* VAL) {

    // Initialization
    double solTime,*val;
    INTEGER4 i, fileformat, dIsDouble, vIsDouble, zoneType, strandID, parentZn, isBlock, debug;
    INTEGER4 iCellMax, jCellMax, kCellMax, nFConns, fNMode, shrConn, fileType;
    INTEGER4 nNodes, nCells, nFaces, connectivityCount, index;
    int *valueLocation = new int[valnum];
    //
    for (int i = 0; i < valnum; i++) {
        valueLocation[i] = 1;
    }
    
    
    /*
     * Open the file and write the tecplot datafile
     * header information
     */
    fileformat = 0;
    debug = 0;
    solTime = 360.0;
    vIsDouble = 1;
    dIsDouble = 1;
    nNodes = Nnum;
    nCells = Enum;
    nFaces = 6; /* Not used */
    zoneType = 5;      /* Brick */
    strandID = 0;     /* StaticZone */
    parentZn = 0;      /* No Parent */
    isBlock = 1;      /* Block */
    iCellMax = 0;
    jCellMax = 0;
    kCellMax = 0;
    nFConns = 0;
    fNMode = 0;
    shrConn = 0;
    fileType = FULL;
    connectivityCount = 8 * Enum;

    i = TECINI142(Title, VariableNames, FileName, ScratchDir,
        &fileformat,
        &fileType,
        &debug,
        &vIsDouble);

    /*
     * Write the zone header information.
     */
    i = TECZNE142((char*)"Simple Zone",
        &zoneType,
        &nNodes,
        &nCells,
        &nFaces,
        &iCellMax,
        &jCellMax,
        &kCellMax,
        &solTime,
        &strandID,
        &parentZn,
        &isBlock,
        &nFConns,
        &fNMode,
        0,              /* TotalNumFaceNodes */
        0,              /* NumConnectedBoundaryFaces */
        0,              /* TotalNumBoundaryConnections */
        NULL,           /* PassiveVarList */
        valueLocation,  /* ValueLocation = Nodal */
        NULL,           /* SharVarFromZone */
        &shrConn);

    /*
    * Write out the field data.
    */
    val = (double*)malloc(nNodes*sizeof(double));
    for (int j = 0; j < valnum; j++) {
        for (int k = 0; k < Nnum; k++) {
            val[k] = VAL[j * Nnum + k];
        }
        i = TECDAT142(&nNodes, val, &dIsDouble);
    }
    free(val);

    i = TECNODE142(&connectivityCount, connect);

    i = TECEND142();

}

