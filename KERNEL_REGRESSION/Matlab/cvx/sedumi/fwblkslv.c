/*
   y = fwblkslv(L,b, [y])
   Given block sparse Cholesky structure L, as generated by
   SPARCHOL, this solves the equation   "L.L * y = b(L.perm,:)",
   i.e. y = L.L\b(L.perm,:).  The diagonal of L.L is taken to
   be all-1, i.e. it uses eye(n) + tril(L.L,-1).

   If b is SPARSE, then the 3rd argument (y) must give the sparsity
   structure of the output variable y.

% This file is part of SeDuMi 1.1 by Imre Polik and Oleksandr Romanko
% Copyright (C) 2005 McMaster University, Hamilton, CANADA  (since 1.1)
%
% Copyright (C) 2001 Jos F. Sturm (up to 1.05R5)
%   Dept. Econometrics & O.R., Tilburg University, the Netherlands.
%   Supported by the Netherlands Organization for Scientific Research (NWO).
%
% Affiliation SeDuMi 1.03 and 1.04Beta (2000):
%   Dept. Quantitative Economics, Maastricht University, the Netherlands.
%
% Affiliations up to SeDuMi 1.02 (AUG1998):
%   CRL, McMaster University, Canada.
%   Supported by the Netherlands Organization for Scientific Research (NWO).
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc.,  51 Franklin Street, Fifth Floor, Boston, MA
% 02110-1301, USA

*/

#include <string.h>
#include "mex.h"
#include "blksdp.h"

#define Y_OUT plhs[0]
#define NPAROUT 1

#define L_IN prhs[0]
#define B_IN prhs[1]
#define MINNPARIN 2
#define Y_IN prhs[2]
#define NPARIN 3

/*typedef struct{
 const double *pr, *pi;
 const mwIndex *jc, *ir;
    } jcir;*/

/* ============================================================
   FORWARD SOLVE:
   ============================================================ */
/* ************************************************************
   PROCEDURE fwsolve -- Solve ynew from L*y = yold, where
     L is lower-triangular.
   INPUT
     L - sparse lower triangular matrix
     xsuper - starting column in L for each (dense) supernode.
     nsuper - number of super nodes
   UPDATED
     y - full vector, on input y = rhs, on output y = L\rhs.
   WORK
     fwork - length max(collen[i] - superlen[i]) <= m-1, where
       collen[i] := L.jc[xsuper[i]+1]-L.jc[xsuper[i]] and
       superlen[i] := xsuper[i+1]-xsuper[i].
   ************************************************************ */
void fwsolve(double *y, const mwIndex *Ljc, const mwIndex *Lir, const double *Lpr,
             const mwIndex *xsuper, const mwIndex nsuper, double *fwork)
{
  mwIndex jsup,i,j,inz,jnnz;
  double yi,yj;

  /* ------------------------------------------------------------
     For each supernode jsup:
     ------------------------------------------------------------ */
  j = xsuper[0];           /* 1st col of current snode (j=0)*/
  inz = Ljc[0];           /* 1st nonzero in L (inz = 0) */
  for(jsup = 1; jsup <= nsuper; jsup++){
/* ------------------------------------------------------------
   The first equation, 1*y=b(j), yields y(j) = b(j).
   ------------------------------------------------------------ */
    mxAssert(inz == Ljc[j],"");
    yj = y[j++];
    ++inz;             /* jump over diagonal entry */
    if(j >= xsuper[jsup])
/* ------------------------------------------------------------
   If supernode is singleton, then simply set y(j+1:m)-=yj*L(j+1:m,j)
   ------------------------------------------------------------ */
      for(; inz < Ljc[j]; inz++)
	y[Lir[inz]] -= yj * Lpr[inz];
    else{
/* ------------------------------------------------------------
   Supernode contains multiple subnodes:
   Remember (i,yi) = 1st subnode, then
   perform dense forward solve within current supernode.
   ------------------------------------------------------------ */
      i = j;
      yi = yj;
      do{
        subscalarmul(y+j, yj, Lpr+inz, xsuper[jsup] - j);
        inz = Ljc[j];
        yj = y[j++];
        ++inz;             /* jump over diagonal entry */
      } while(j < xsuper[jsup]);
      jnnz = Ljc[j] - inz;
/* ------------------------------------------------------------
   jnnz = number of later entries that are influenced by this supernode.
   Compute the update in the array fwork(jnnz)
   ------------------------------------------------------------ */
      if(jnnz > 0){
        scalarmul(fwork, yj, Lpr+inz,jnnz);
        while(i < j){
          addscalarmul(fwork,yi,Lpr+Ljc[i]-jnnz,jnnz);
          yi = y[i++];
        }
/* ------------------------------------------------------------
   Update y with fwork at the specified sparse locations
   ------------------------------------------------------------ */
        for(i = 0; i < jnnz; i++)
          y[Lir[inz++]] -= fwork[i];
      }
    }
  }
}

/* ************************************************************
   PROCEDURE selfwsolve -- Solve ynew from L*y = yold, where
     L is lower-triangular and y is SPARSE.
   INPUT
     L     - sparse lower triangular matrix
     xsuper - length nsuper+1, start of each (dense) supernode.
     nsuper - number of super nodes
     snode - length m array, mapping each node to the supernode containing it.
     yir   - length ynnz array, listing all possible nonzeros entries in y.
     ynnz  - number of nonzeros in y (from symbfwslv).
   UPDATED
     y - full vector, on input y = rhs, on output y = L\rhs.
        only the yir(0:ynnz-1) entries are used and defined.
   ************************************************************ */
void selfwsolve(double *y, const mwIndex *Ljc, const mwIndex *Lir, const double *Lpr,
                const mwIndex *xsuper, const mwIndex nsuper,
                const mwIndex *snode, const mwIndex *yir, const mwIndex ynnz)
{
  mwIndex jsup,j,inz,jnz;
  double yj;

  if(ynnz <= 0)
    return;
/* ------------------------------------------------------------
   Forward solve on each nonzero supernode snode[yir[jnz]] (=jsup-1).
   ------------------------------------------------------------ */
  jnz = 0;
  while(jnz < ynnz){
    j = yir[jnz];
    jsup = snode[j] + 1;
    jnz += xsuper[jsup] - j;          /* point to next nonzero supernode */
    while(j < xsuper[jsup]){
/* ------------------------------------------------------------
   Do dense computations on supernode.
   The first equation, 1*y=b(j), yields y(j) = b(j).
   ------------------------------------------------------------ */
      inz = Ljc[j];
      yj = y[j++];
      ++inz;             /* jump over diagonal entry */
/* ------------------------------------------------------------
   Forward solution: y(j+1:m) -= yj * L(j+1:m,j)
   ------------------------------------------------------------ */
      subscalarmul(y+j, yj, Lpr+inz, xsuper[jsup] - j);
      for(inz += xsuper[jsup] - j; inz < Ljc[j]; inz++)
	y[Lir[inz]] -= yj * Lpr[inz];
    }
  }
}

/* ============================================================
   MAIN: MEXFUNCTION
   ============================================================ */
/* ************************************************************
   PROCEDURE mexFunction - Entry for Matlab
   y = fwblksolve(L,b, [y])
     y = L.L \ b(L.perm)
   ************************************************************ */
void mexFunction(const int nlhs, mxArray *plhs[],
  const int nrhs, const mxArray *prhs[])
{
 const mxArray *L_FIELD;
 mwIndex m,n, j, k, nsuper, inz;
 double *y,*fwork;
 const double *permPr, *b, *xsuperPr;
 const mwIndex *yjc, *yir, *bjc, *bir;
 mwIndex *perm, *invperm, *snode, *xsuper, *iwork;
 jcir L;
 char bissparse;
 /* ------------------------------------------------------------
    Check for proper number of arguments 
    ------------------------------------------------------------ */
 mxAssert(nrhs >= MINNPARIN, "fwblkslv requires more input arguments.");
 mxAssert(nlhs <= NPAROUT, "fwblkslv generates only 1 output argument.");
 /* ------------------------------------------------------------
    Disassemble block Cholesky structure L
    ------------------------------------------------------------ */
 mxAssert(mxIsStruct(L_IN), "Parameter `L' should be a structure.");
 if( (L_FIELD = mxGetField(L_IN,(mwIndex)0,"perm")) == NULL)      /* L.perm */
   mexErrMsgTxt("Missing field L.perm.");
 m = mxGetM(L_FIELD) * mxGetN(L_FIELD);
 permPr = mxGetPr(L_FIELD);
 if( (L_FIELD = mxGetField(L_IN,(mwIndex)0,"L")) == NULL)      /* L.L */
   mexErrMsgTxt("Missing field L.L.");
 if( m != mxGetM(L_FIELD) || m != mxGetN(L_FIELD) )
   mexErrMsgTxt("Size L.L mismatch.");
 if(!mxIsSparse(L_FIELD))
   mexErrMsgTxt("L.L should be sparse.");
 L.jc = mxGetJc(L_FIELD);
 L.ir = mxGetIr(L_FIELD);
 L.pr = mxGetPr(L_FIELD);
 if( (L_FIELD = mxGetField(L_IN,(mwIndex)0,"xsuper")) == NULL)      /* L.xsuper */
   mexErrMsgTxt("Missing field L.xsuper.");
 nsuper = mxGetM(L_FIELD) * mxGetN(L_FIELD) - 1;
 if( nsuper > m )
   mexErrMsgTxt("Size L.xsuper mismatch.");
 xsuperPr = mxGetPr(L_FIELD);
 /* ------------------------------------------------------------
    Get rhs matrix b.
    If it is sparse, then we also need the sparsity structure of y.
    ------------------------------------------------------------ */
 b = mxGetPr(B_IN);
 if( mxGetM(B_IN) != m )
   mexErrMsgTxt("Size mismatch b.");
 n = mxGetN(B_IN);
 if( (bissparse = mxIsSparse(B_IN)) ){
   bjc = mxGetJc(B_IN);
   bir = mxGetIr(B_IN);
   if(nrhs < NPARIN)
     mexErrMsgTxt("fwblkslv requires more inputs in case of sparse b.");
   if(mxGetM(Y_IN) != m || mxGetN(Y_IN) != n)
     mexErrMsgTxt("Size mismatch y.");
   if(!mxIsSparse(Y_IN))
     mexErrMsgTxt("y should be sparse.");
 }
/* ------------------------------------------------------------
   Allocate output y. If bissparse, then Y_IN gives the sparsity structure.
   ------------------------------------------------------------ */
 if(!bissparse)
   Y_OUT = mxCreateDoubleMatrix(m, n, mxREAL);
 else{
   yjc = mxGetJc(Y_IN);
   yir = mxGetIr(Y_IN);
   Y_OUT = mxCreateSparse(m,n, yjc[n],mxREAL);
   memcpy(mxGetJc(Y_OUT), yjc, (n+1) * sizeof(mwIndex));
   memcpy(mxGetIr(Y_OUT), yir, yjc[n] * sizeof(mwIndex));
 }
 y = mxGetPr(Y_OUT);
/* ------------------------------------------------------------
   Allocate working arrays fwork(m) and iwork(2*m + nsuper+1)
   ------------------------------------------------------------ */
 fwork = (double *) mxCalloc(m, sizeof(double));
 iwork = (mwIndex *) mxCalloc(2*m+nsuper+1, sizeof(mwIndex));
 perm = iwork;
 invperm = perm;
 xsuper = iwork + m;
 snode = xsuper + (nsuper+1);
/* ------------------------------------------------------------
   Convert real to integer array, and from Fortran to C style.
   In case of sparse b, we store the inverse perm, instead of perm itself.
   ------------------------------------------------------------ */
 for(k = 0; k <= nsuper; k++)
   xsuper[k] = xsuperPr[k] - 1;
 if(!bissparse)
   for(k = 0; k < m; k++)               /* Get perm if !bissparse */
     perm[k] = permPr[k] - 1;
 else{
   for(k = 0; k < m; k++){              /* Get invperm if bissparse */
     j = permPr[k];
     invperm[--j] = k;
   }
/* ------------------------------------------------------------
   In case of sparse b, we also create snode, which maps each subnode
   to the supernode containing it.
   ------------------------------------------------------------ */
   for(j = 0, k = 0; k < nsuper; k++)
     while(j < xsuper[k+1])
       snode[j++] = k;
 }
/* ------------------------------------------------------------
   The actual job is done here: y = L\b(perm).
   ------------------------------------------------------------ */
 if(!bissparse)
   for(j = 0; j < n; j++){
     for(k = 0; k < m; k++)            /* y = b(perm) */
       y[k] = b[perm[k]];
     fwsolve(y,L.jc,L.ir,L.pr,xsuper,nsuper,fwork);
     y += m; b += m;
   }
 else
   for(j = 0, inz = 0; j < n; j++){
     for(k = inz; k < yjc[j+1]; k++)            /* fwork = all-0 */
       fwork[yir[k]] = 0.0;
     for(k = bjc[j]; k < bjc[j+1]; k++)            /* fwork = b(perm) */
       fwork[invperm[bir[k]]] = b[k];
     selfwsolve(fwork,L.jc,L.ir,L.pr,xsuper,nsuper, snode,
                yir+inz,yjc[j+1]-inz);
     for(; inz < yjc[j+1]; inz++)
       y[inz] = fwork[yir[inz]];
   }
 /* ------------------------------------------------------------
    RELEASE WORKING ARRAYS.
    ------------------------------------------------------------ */
 mxFree(iwork);
 mxFree(fwork);
}
