#ifndef __STREAM_PREDICATES_HPP__
#define __STREAM_PREDICATES_HPP__

#include "defines.h"
#include <math.h>

constexpr fp_tt const splitter = 4097.0; /* = 2^ceiling(p / 2) + 1.  Used to split floats in half. */
constexpr fp_tt const epsilon = 0.000000059604644775390625; /* = 2^(-p).  Used to estimate roundoff errors. */

#define Fast_Two_Sum_Tail(a, b, x, y) \
  bvirt = x - a; \
  y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
  x = (fp_tt) (a + b); \
  Fast_Two_Sum_Tail(a, b, x, y)

#define Fast_Two_Diff_Tail(a, b, x, y) \
  bvirt = a - x; \
  y = bvirt - b

#define Fast_Two_Diff(a, b, x, y) \
  x = (fp_tt) (a - b); \
  Fast_Two_Diff_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y) \
  bvirt = (fp_tt) (x - a); \
  avirt = x - bvirt; \
  bround = b - bvirt; \
  around = a - avirt; \
  y = around + bround

#define Two_Sum(a, b, x, y) \
  x = (fp_tt) (a + b); \
  Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y) \
  bvirt = (fp_tt) (a - x); \
  avirt = x + bvirt; \
  bround = bvirt - b; \
  around = a - avirt; \
  y = around + bround

#define Two_Diff(a, b, x, y) \
  x = (fp_tt) (a - b); \
  Two_Diff_Tail(a, b, x, y)

#define Split(a, ahi, alo) \
  c = (fp_tt) (splitter * a); \
  abig = (fp_tt) (c - a); \
  ahi = c - abig; \
  alo = a - ahi

#define Two_Product_Tail(a, b, x, y) \
  Split(a, ahi, alo); \
  Split(b, bhi, blo); \
  err1 = x - (ahi * bhi); \
  err2 = err1 - (alo * bhi); \
  err3 = err2 - (ahi * blo); \
  y = (alo * blo) - err3

#define Two_Product(a, b, x, y) \
  x = (fp_tt) (a * b); \
  Two_Product_Tail(a, b, x, y)

/* Two_Product_Presplit() is Two_Product() where one of the inputs has       */
/*   already been split.  Avoids redundant splitting.                        */

#define Two_Product_Presplit(a, b, bhi, blo, x, y) \
  x = (fp_tt) (a * b); \
  Split(a, ahi, alo); \
  err1 = x - (ahi * bhi); \
  err2 = err1 - (alo * bhi); \
  err3 = err2 - (ahi * blo); \
  y = (alo * blo) - err3

/* Two_Product_2Presplit() is Two_Product() where both of the inputs have    */
/*   already been split.  Avoids redundant splitting.                        */

#define Two_Product_2Presplit(a, ahi, alo, b, bhi, blo, x, y) \
  x = (fp_tt) (a * b); \
  err1 = x - (ahi * bhi); \
  err2 = err1 - (alo * bhi); \
  err3 = err2 - (ahi * blo); \
  y = (alo * blo) - err3

/* Square() can be done more quickly than Two_Product().                     */

#define Square_Tail(a, x, y) \
  Split(a, ahi, alo); \
  err1 = x - (ahi * ahi); \
  err3 = err1 - ((ahi + ahi) * alo); \
  y = (alo * alo) - err3

#define Square(a, x, y) \
  x = (fp_tt) (a * a); \
  Square_Tail(a, x, y)

/* Macros for summing expansions of various fixed lengths.  These are all    */
/*   unrolled versions of Expansion_Sum().                                   */

#define Two_One_Sum(a1, a0, b, x2, x1, x0) \
  Two_Sum(a0, b , _i, x0); \
  Two_Sum(a1, _i, x2, x1)

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
  Two_Diff(a0, b , _i, x0); \
  Two_Sum( a1, _i, x2, x1)

#define Two_Two_Sum(a1, a0, b1, b0, x3, x2, x1, x0) \
  Two_One_Sum(a1, a0, b0, _j, _0, x0); \
  Two_One_Sum(_j, _0, b1, x3, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
  Two_One_Diff(a1, a0, b0, _j, _0, x0); \
  Two_One_Diff(_j, _0, b1, x3, x2, x1)

#define Four_One_Sum(a3, a2, a1, a0, b, x4, x3, x2, x1, x0) \
  Two_One_Sum(a1, a0, b , _j, x1, x0); \
  Two_One_Sum(a3, a2, _j, x4, x3, x2)

#define Four_Two_Sum(a3, a2, a1, a0, b1, b0, x5, x4, x3, x2, x1, x0) \
  Four_One_Sum(a3, a2, a1, a0, b0, _k, _2, _1, _0, x0); \
  Four_One_Sum(_k, _2, _1, _0, b1, x5, x4, x3, x2, x1)

#define Four_Four_Sum(a3, a2, a1, a0, b4, b3, b1, b0, x7, x6, x5, x4, x3, x2, \
                      x1, x0) \
  Four_Two_Sum(a3, a2, a1, a0, b1, b0, _l, _2, _1, _0, x1, x0); \
  Four_Two_Sum(_l, _2, _1, _0, b4, b3, x7, x6, x5, x4, x3, x2)

#define Eight_One_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b, x8, x7, x6, x5, x4, \
                      x3, x2, x1, x0) \
  Four_One_Sum(a3, a2, a1, a0, b , _j, x3, x2, x1, x0); \
  Four_One_Sum(a7, a6, a5, a4, _j, x8, x7, x6, x5, x4)

#define Eight_Two_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b1, b0, x9, x8, x7, \
                      x6, x5, x4, x3, x2, x1, x0) \
  Eight_One_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b0, _k, _6, _5, _4, _3, _2, \
                _1, _0, x0); \
  Eight_One_Sum(_k, _6, _5, _4, _3, _2, _1, _0, b1, x9, x8, x7, x6, x5, x4, \
                x3, x2, x1)

#define Eight_Four_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b4, b3, b1, b0, x11, \
                       x10, x9, x8, x7, x6, x5, x4, x3, x2, x1, x0) \
  Eight_Two_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b1, b0, _l, _6, _5, _4, _3, \
                _2, _1, _0, x1, x0); \
  Eight_Two_Sum(_l, _6, _5, _4, _3, _2, _1, _0, b4, b3, x11, x10, x9, x8, \
                x7, x6, x5, x4, x3, x2)

/* Macros for multiplying expansions of various fixed lengths.               */

#define Two_One_Product(a1, a0, b, x3, x2, x1, x0) \
  Split(b, bhi, blo); \
  Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
  Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, x1); \
  Fast_Two_Sum(_j, _k, x3, x2)

#define Four_One_Product(a3, a2, a1, a0, b, x7, x6, x5, x4, x3, x2, x1, x0) \
  Split(b, bhi, blo); \
  Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
  Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, x1); \
  Fast_Two_Sum(_j, _k, _i, x2); \
  Two_Product_Presplit(a2, b, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, x3); \
  Fast_Two_Sum(_j, _k, _i, x4); \
  Two_Product_Presplit(a3, b, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, x5); \
  Fast_Two_Sum(_j, _k, x7, x6)

#define Two_Two_Product(a1, a0, b1, b0, x7, x6, x5, x4, x3, x2, x1, x0) \
  Split(a0, a0hi, a0lo); \
  Split(b0, bhi, blo); \
  Two_Product_2Presplit(a0, a0hi, a0lo, b0, bhi, blo, _i, x0); \
  Split(a1, a1hi, a1lo); \
  Two_Product_2Presplit(a1, a1hi, a1lo, b0, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, _1); \
  Fast_Two_Sum(_j, _k, _l, _2); \
  Split(b1, bhi, blo); \
  Two_Product_2Presplit(a0, a0hi, a0lo, b1, bhi, blo, _i, _0); \
  Two_Sum(_1, _0, _k, x1); \
  Two_Sum(_2, _k, _j, _1); \
  Two_Sum(_l, _j, _m, _2); \
  Two_Product_2Presplit(a1, a1hi, a1lo, b1, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _n, _0); \
  Two_Sum(_1, _0, _i, x2); \
  Two_Sum(_2, _i, _k, _1); \
  Two_Sum(_m, _k, _l, _2); \
  Two_Sum(_j, _n, _k, _0); \
  Two_Sum(_1, _0, _j, x3); \
  Two_Sum(_2, _j, _i, _1); \
  Two_Sum(_l, _i, _m, _2); \
  Two_Sum(_1, _k, _i, x4); \
  Two_Sum(_2, _i, _k, x5); \
  Two_Sum(_m, _k, x7, x6)

/* An expansion of length two can be squared more quickly than finding the   */
/*   product of two different expansions of length two, and the result is    */
/*   guaranteed to have no more than six (rather than eight) components.     */

#define Two_Square(a1, a0, x5, x4, x3, x2, x1, x0) \
  Square(a0, _j, x0); \
  _0 = a0 + a0; \
  Two_Product(a1, _0, _k, _1); \
  Two_One_Sum(_k, _1, _j, _l, _2, x1); \
  Square(a1, _j, _1); \
  Two_Two_Sum(_j, _1, _l, _2, x5, x4, x3, x2)

__host__ __device__ static int fast_expansion_sum_zeroelim(
    int elen, fp_tt* e, int flen, fp_tt* f, fp_tt* h) { /* h cannot be e or f. */
    fp_tt Q;
    fp_tt Qnew;
    fp_tt hh;
    fp_tt bvirt;
    fp_tt avirt, bround, around;
    int   eindex, findex, hindex;
    fp_tt enow, fnow;

    enow   = e[0];
    fnow   = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q    = enow;
        enow = e[++eindex];
    } else {
        Q    = fnow;
        fnow = f[++findex];
    }
    hindex = 0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            enow = e[++eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            fnow = f[++findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                enow = e[++eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                fnow = f[++findex];
            }
            Q = Qnew;
            if (hh != 0.0) {
                h[hindex++] = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
        Q    = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
        Q    = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

__host__ __device__ static int scale_expansion_zeroelim(
    int elen, fp_tt* e, fp_tt b, fp_tt* h) { /* e and h cannot be the same. */
    fp_tt Q, sum;
    fp_tt hh;
    fp_tt product1;
    fp_tt product0;
    int   eindex, hindex;
    fp_tt enow;
    fp_tt bvirt;
    fp_tt avirt, bround, around;
    fp_tt c;
    fp_tt abig;
    fp_tt ahi, alo, bhi, blo;
    fp_tt err1, err2, err3;

    Split(b, bhi, blo);
    Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
    hindex = 0;
    if (hh != 0) {
        h[hindex++] = hh;
    }
    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
        Two_Sum(Q, product0, sum, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
        Fast_Two_Sum(product1, sum, Q, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

// ================================ 2D ========================================

constexpr fp_tt iccerrboundA = (10.0 + 96.0 * epsilon) * epsilon; // Incircle predicate bound
constexpr fp_tt ccwerrboundA = (3.0 + 16.0 * epsilon) * epsilon; // orient2d predicate bound

__host__ __device__ static fp_tt orient2dexact(
        fp_tt const * const __restrict__ pa, 
        fp_tt const * const __restrict__ pb, 
        fp_tt const * const __restrict__ pc
        ) {
  fp_tt axby1, axcy1;
  fp_tt axby0, axcy0;
  fp_tt aterms[4], bterms[4];
  fp_tt v[8], w[12];
  int vlength, wlength;

  fp_tt bvirt;
  fp_tt avirt, bround, around;
  fp_tt c;
  fp_tt abig;
  fp_tt ahi, alo, bhi, blo;
  fp_tt err1, err2, err3;
  fp_tt _i, _j;
  fp_tt _0;

  Two_Product(pa[0], pb[1], axby1, axby0);   // ax*by
  Two_Product(pa[0], pc[1], axcy1, axcy0);   // ax*cy
  Two_Two_Diff(axby1, axby0, axcy1, axcy0, aterms[3], aterms[2], aterms[1], aterms[0]); // ax*by - ax*cy

  Two_Product(pb[0], pc[1], axby1, axby0);   // bx*cy
  Two_Product(pb[0], pa[1], axcy1, axcy0);   // bx*ay
  Two_Two_Diff(axby1, axby0, axcy1, axcy0, bterms[3], bterms[2], bterms[1], bterms[0]); // bx*cy - bx*ay
                                                                                        
  // ax*by - ax*cy + bx*cy - bx*ay
  vlength = fast_expansion_sum_zeroelim(4, aterms, 4, bterms, v); 

  Two_Product(pc[0], pa[1], axby1, axby0);   // cx*ay
  Two_Product(pc[0], pb[1], axcy1, axcy0);   // cx*by
  Two_Two_Diff(axby1, axby0, axcy1, axcy0, aterms[3], aterms[2], aterms[1], aterms[0]); // cx*ay - cx*by

  // ax*by - ax*cy + bx*cy - bx*ay + cx*ay - cx*by
  // = ax*(by - cy) + bx*(cy - ay) + cx*(ay - by)
  wlength = fast_expansion_sum_zeroelim(vlength, v, 4, aterms, w);

  return w[wlength - 1];
}


__host__ __device__ fp_tt static orient2d(
        fp_tt const * const __restrict__ pa, 
        fp_tt const * const __restrict__ pb, 
        fp_tt const * const __restrict__ pc) {
  fp_tt detleft, detright, det;
  fp_tt detsum, errbound;

  detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
  detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
  det = detleft - detright;

  if (detleft > 0.0f) {
    if (detright <= 0.0f) {
      return det;
    } else {
      detsum = detleft + detright;
    }
  } else if (detleft < 0.0f) {
    if (detright >= 0.0f) {
      return det;
    } else {
      detsum = -detleft - detright;
    }
  } else {
    return det;
  }

  errbound = ccwerrboundA * detsum;
  if ((det >= errbound) || (-det >= errbound)) {
    return det;
  }

  return orient2dexact(pa, pb, pc);
}

__host__ __device__ fp_tt static incircleexact(
        fp_tt const * const __restrict__ pa,
        fp_tt const * const __restrict__ pb,
        fp_tt const * const __restrict__ pc,
        fp_tt const * const __restrict__ pd) {
    fp_tt axby1, bxcy1, cxdy1, dxay1, axcy1, bxdy1;
    fp_tt bxay1, cxby1, dxcy1, axdy1, cxay1, dxby1;
    fp_tt axby0, bxcy0, cxdy0, dxay0, axcy0, bxdy0;
    fp_tt bxay0, cxby0, dxcy0, axdy0, cxay0, dxby0;
    fp_tt ab[5], bc[5], cd[5], da[5], ac[5], bd[5];
    fp_tt temp8[8];
    int   templen;
    fp_tt abc[12], bcd[12], cda[12], dab[12];
    int   abclen, bcdlen, cdalen, dablen;
    fp_tt det24x[24], det24y[24], det48x[48], det48y[48];
    int   xlen, ylen;
    fp_tt adet[96], bdet[96]; 
    int   alen, blen, clen, dlen;
    fp_tt abdet[192], cddet[192];
    int   ablen, cdlen;
    fp_tt deter[384];
    int   deterlen;
    int   i;

    // Used by the macros for computing approximate operation and error
    fp_tt bvirt;
    fp_tt avirt, bround, around;
    fp_tt c;
    fp_tt abig;
    fp_tt ahi, alo, bhi, blo;
    fp_tt err1, err2, err3;
    fp_tt _i, _j;
    fp_tt _0;

    // minor(a, b)
    Two_Product(pa[0], pb[1], axby1, axby0); // ax*by
    Two_Product(pb[0], pa[1], bxay1, bxay0); // bx*ay
    Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]); // ab = ax*by - bx*ay

    // minor(b, c)
    Two_Product(pb[0], pc[1], bxcy1, bxcy0); // bx*cy
    Two_Product(pc[0], pb[1], cxby1, cxby0); // cx*by
    Two_Two_Diff(bxcy1, bxcy0, cxby1, cxby0, bc[3], bc[2], bc[1], bc[0]); // bc = bx*cy - cx*by

    // minor(c, d)
    Two_Product(pc[0], pd[1], cxdy1, cxdy0); // cx*dy
    Two_Product(pd[0], pc[1], dxcy1, dxcy0); // dx*cy
    Two_Two_Diff(cxdy1, cxdy0, dxcy1, dxcy0, cd[3], cd[2], cd[1], cd[0]); // cd = cx*dy - dx*cy

    // minor(d, a) = - minor(a, d)
    Two_Product(pd[0], pa[1], dxay1, dxay0); // dx*ay
    Two_Product(pa[0], pd[1], axdy1, axdy0); // ax*dy
    Two_Two_Diff(dxay1, dxay0, axdy1, axdy0, da[3], da[2], da[1], da[0]); // da = dx*ay - axdy

    // minor(a, c)
    Two_Product(pa[0], pc[1], axcy1, axcy0); // ax*cy
    Two_Product(pc[0], pa[1], cxay1, cxay0); // cx*ay
    Two_Two_Diff(axcy1, axcy0, cxay1, cxay0, ac[3], ac[2], ac[1], ac[0]); // ac = ax*cy - cx*ay

    // minor(b, d)
    Two_Product(pb[0], pd[1], bxdy1, bxdy0); // bx*dy
    Two_Product(pd[0], pb[1], dxby1, dxby0); // dx*by
    Two_Two_Diff(bxdy1, bxdy0, dxby1, dxby0, bd[3], bd[2], bd[1], bd[0]); // bc = bx*dy - dx*by

    // minor(c, d, a) = minor(a, c, d)
    templen = fast_expansion_sum_zeroelim(4, cd, 4, da, temp8);        // cd + da
    cdalen  = fast_expansion_sum_zeroelim(templen, temp8, 4, ac, cda); // cda = cd - ad + ac = cd + da + ac
                                                                       
    // minor(d, a, b) = minor(a, b, d)
    templen = fast_expansion_sum_zeroelim(4, da, 4, ab, temp8);        // da + ab
    dablen  = fast_expansion_sum_zeroelim(templen, temp8, 4, bd, dab); // dab = ab - ad + bd = ab + da + bd 

    // Change signs of the minors for the computations of abd/bcd
    for (i = 0; i < 4; i++) {
        bd[i] = -bd[i];  // bd => db
        ac[i] = -ac[i];  // ac => ca
    }

    templen = fast_expansion_sum_zeroelim(4, ab, 4, bc, temp8);        // ab + bc
    abclen  = fast_expansion_sum_zeroelim(templen, temp8, 4, ac, abc); // abc = ab - ac + bc = ab + ca + bc 

    templen = fast_expansion_sum_zeroelim(4, bc, 4, cd, temp8);        // bc + cd
    bcdlen  = fast_expansion_sum_zeroelim(templen, temp8, 4, bd, bcd); // bcd = bc - bd + cd =  bc + db + cd

    // adet = alift*minor(b, c, d)
    xlen = scale_expansion_zeroelim(bcdlen, bcd, pa[0], det24x);    // adetx = bcd * ax
    xlen = scale_expansion_zeroelim(xlen, det24x, pa[0], det48x);   // adetx2 = adetx * ax
    ylen = scale_expansion_zeroelim(bcdlen, bcd, pa[1], det24y);    // adety = bcd * ay
    ylen = scale_expansion_zeroelim(ylen, det24y, pa[1], det48y);   // adety2 = adety * ay
    alen = fast_expansion_sum_zeroelim(xlen, det48x, ylen, det48y, adet); // adet = adetx2 + adety2 = bcd * (ax*ax + ay*ay) = bcd * alift

    // bdet = blift*minor(a, c, d)
    xlen = scale_expansion_zeroelim(cdalen, cda, pb[0], det24x);    // bdetx = cda * bx
    xlen = scale_expansion_zeroelim(xlen, det24x, -pb[0], det48x);  // bdetx2 = - bdetx * bx
    ylen = scale_expansion_zeroelim(cdalen, cda, pb[1], det24y);    // bdety = cda * by
    ylen = scale_expansion_zeroelim(ylen, det24y, -pb[1], det48y);  // bdety2 = - bdety * by
    blen = fast_expansion_sum_zeroelim(xlen, det48x, ylen, det48y, bdet); // bdet = bdetx2 + bdety2 = - cda * (bx*bx + by*by) = - cda * blift
    

    ablen    = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet); // adet + bdet

    fp_tt* const __restrict__ cdet = adet;
    fp_tt* const __restrict__ ddet = bdet;

    // cdet = clift*minor(a, b, d)
    xlen = scale_expansion_zeroelim(dablen, dab, pc[0], det24x);    // cdetx = dab * cx
    xlen = scale_expansion_zeroelim(xlen, det24x, pc[0], det48x);   // cdetx2 = cdetx * cx
    ylen = scale_expansion_zeroelim(dablen, dab, pc[1], det24y);    // cdety = dab * cy
    ylen = scale_expansion_zeroelim(ylen, det24y, pc[1], det48y);   // cdety2 = cdety * cy
    clen = fast_expansion_sum_zeroelim(xlen, det48x, ylen, det48y, cdet);  // cdet = cdetx2 + cdety2 = dab * (cx*cx + cy*cy) = dab * clift

    // ddet = dlift*minor(a, b, c)
    xlen = scale_expansion_zeroelim(abclen, abc, pd[0], det24x);    // ddetx = abc*dx
    xlen = scale_expansion_zeroelim(xlen, det24x, -pd[0], det48x);  // ddetx2 = - ddetx*dx
    ylen = scale_expansion_zeroelim(abclen, abc, pd[1], det24y);    // ddety = abc*dy
    ylen = scale_expansion_zeroelim(ylen, det24y, -pd[1], det48y);  // ddety2 = - ddety*dy
    dlen = fast_expansion_sum_zeroelim(xlen, det48x, ylen, det48y, ddet); // ddet = ddetx2 + ddety2 = - abc * (dx*dx + dy*dy) = - abd * dlift

    cdlen    = fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet); // cdet + ddet
    deterlen = fast_expansion_sum_zeroelim(ablen, abdet, cdlen, cddet, deter); // adet + bdet + cdet + ddet

    return deter[deterlen - 1]; // get most important component
}

__host__ __device__ fp_tt static incircle(
        fp_tt const * const __restrict__ pa,
        fp_tt const * const __restrict__ pb,
        fp_tt const * const __restrict__ pc,
        fp_tt const * const __restrict__ pd,
        fp_tt       * const __restrict__ err) {
    fp_tt adx, bdx, cdx, ady, bdy, cdy;
    fp_tt bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    fp_tt alift, blift, clift;
    fp_tt det;
    fp_tt permanent;

    adx = pa[0] - pd[0];
    bdx = pb[0] - pd[0];
    cdx = pc[0] - pd[0];
    ady = pa[1] - pd[1];
    bdy = pb[1] - pd[1];
    cdy = pc[1] - pd[1];

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;
    alift  = fmaf(adx, adx, ady * ady);

    cdxady = cdx * ady;
    adxcdy = adx * cdy;
    blift  = fmaf(bdx, bdx, bdy * bdy);

    adxbdy = adx * bdy;
    bdxady = bdx * ady;
    clift  = fmaf(cdx, cdx, cdy * cdy);

    det = fmaf(alift, (bdxcdy - cdxbdy), 
            fmaf(blift, (cdxady - adxcdy), clift * (adxbdy - bdxady)));

    permanent = fmaf(alift, (fabsf(bdxcdy) + fabsf(cdxbdy)),
                fmaf(blift, (fabsf(cdxady) + fabsf(adxcdy)),
                (fabsf(adxbdy) + fabsf(bdxady)) * clift));
    *err = iccerrboundA * permanent;
    return det;
}


// =================================== 3D =====================================

constexpr fp_tt isperrboundA   = (16.0 + 224.0 * epsilon) * epsilon; // Insphere predicate bound
constexpr fp_tt o3derrboundA = (7.0 + 56.0 * epsilon) * epsilon; // orient3d predicate bound

__host__ __device__ static fp_tt orient3dexact(
        fp_tt const * const __restrict__ pa, 
        fp_tt const * const __restrict__ pb,
        fp_tt const * const __restrict__ pc,
        fp_tt const * const __restrict__ pd
    ) {
  fp_tt axby1, bxcy1, cxdy1, dxay1, axcy1, bxdy1;
  fp_tt bxay1, cxby1, dxcy1, axdy1, cxay1, dxby1;
  fp_tt axby0, bxcy0, cxdy0, dxay0, axcy0, bxdy0;
  fp_tt bxay0, cxby0, dxcy0, axdy0, cxay0, dxby0;
  fp_tt ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
  fp_tt temp8[8];
  int templen;
  fp_tt abc[12], bcd[12], cda[12], dab[12];
  int abclen, bcdlen, cdalen, dablen;
  fp_tt adet[24], bdet[24], cdet[24], ddet[24];
  int alen, blen, clen, dlen;
  fp_tt abdet[48], cddet[48];
  int ablen, cdlen;
  fp_tt deter[96];
  int deterlen;
  int i;

  fp_tt bvirt;
  fp_tt avirt, bround, around;
  fp_tt c;
  fp_tt abig;
  fp_tt ahi, alo, bhi, blo;
  fp_tt err1, err2, err3;
  fp_tt _i, _j;
  fp_tt _0;

  Two_Product(pa[0], pb[1], axby1, axby0); // ax*by
  Two_Product(pb[0], pa[1], bxay1, bxay0); // bx*ay
  Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]); // ab = ax*by - bx*ay

  Two_Product(pb[0], pc[1], bxcy1, bxcy0); // bx*cy 
  Two_Product(pc[0], pb[1], cxby1, cxby0); // cx*by
  Two_Two_Diff(bxcy1, bxcy0, cxby1, cxby0, bc[3], bc[2], bc[1], bc[0]); // bc = bx*cy - cx*by

  Two_Product(pc[0], pd[1], cxdy1, cxdy0); // cx*dy 
  Two_Product(pd[0], pc[1], dxcy1, dxcy0); // dx*cy
  Two_Two_Diff(cxdy1, cxdy0, dxcy1, dxcy0, cd[3], cd[2], cd[1], cd[0]); // cd = cx*dy - dx*cy

  Two_Product(pd[0], pa[1], dxay1, dxay0); // dx*ay 
  Two_Product(pa[0], pd[1], axdy1, axdy0); // ax*dy
  Two_Two_Diff(dxay1, dxay0, axdy1, axdy0, da[3], da[2], da[1], da[0]); // da = dx*ay - ax*dy

  Two_Product(pa[0], pc[1], axcy1, axcy0); // ax*cy 
  Two_Product(pc[0], pa[1], cxay1, cxay0); // cx*ay
  Two_Two_Diff(axcy1, axcy0, cxay1, cxay0, ac[3], ac[2], ac[1], ac[0]); // ac = ax*cy - cx*ay

  Two_Product(pb[0], pd[1], bxdy1, bxdy0); // bx*dy 
  Two_Product(pd[0], pb[1], dxby1, dxby0); // dx*ay
  Two_Two_Diff(bxdy1, bxdy0, dxby1, dxby0, bd[3], bd[2], bd[1], bd[0]); // bd = bx*dy - dx*ay

  templen = fast_expansion_sum_zeroelim(4, cd, 4, da, temp8);          // cd + da
  cdalen = fast_expansion_sum_zeroelim(templen, temp8, 4, ac, cda);    // cda = cd + da + ac
  templen = fast_expansion_sum_zeroelim(4, da, 4, ab, temp8);          // da + ab
  dablen = fast_expansion_sum_zeroelim(templen, temp8, 4, bd, dab);    // dab = da + ab + bd
  for (i = 0; i < 4; i++) {
    bd[i] = -bd[i]; // bd => -db
    ac[i] = -ac[i]; // ac => -ca
  }
  templen = fast_expansion_sum_zeroelim(4, ab, 4, bc, temp8);          // ab + bc
  abclen = fast_expansion_sum_zeroelim(templen, temp8, 4, ac, abc);    // abc = ab + bc + ca
  templen = fast_expansion_sum_zeroelim(4, bc, 4, cd, temp8);          // bc + cd
  bcdlen = fast_expansion_sum_zeroelim(templen, temp8, 4, bd, bcd);    // bcd = bc + cd + bd

  alen = scale_expansion_zeroelim(bcdlen, bcd, pa[2], adet);  // adet =  az*bcd
  blen = scale_expansion_zeroelim(cdalen, cda, -pb[2], bdet); // bdet = -bz*cda
  clen = scale_expansion_zeroelim(dablen, dab, pc[2], cdet);  // cdet =  cz*dab
  dlen = scale_expansion_zeroelim(abclen, abc, -pd[2], ddet); // ddet = -dz*abd

  ablen = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet); // abdet = adet + bdet
  cdlen = fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet); // cddet = cdet + ddet
  deterlen = fast_expansion_sum_zeroelim(ablen, abdet, cdlen, cddet, deter); // deter = abdet + cddet

  return deter[deterlen - 1];
}
__host__ __device__ static fp_tt orient3d(
        fp_tt const * const __restrict__ pa,
        fp_tt const * const __restrict__ pb,
        fp_tt const * const __restrict__ pc,
        fp_tt const * const __restrict__ pd
    ) {
  fp_tt adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz;
  fp_tt bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
  fp_tt det;
  fp_tt permanent, errbound;

  adx = pa[0] - pd[0];
  bdx = pb[0] - pd[0];
  cdx = pc[0] - pd[0];
  ady = pa[1] - pd[1];
  bdy = pb[1] - pd[1];
  cdy = pc[1] - pd[1];
  adz = pa[2] - pd[2];
  bdz = pb[2] - pd[2];
  cdz = pc[2] - pd[2];

  bdxcdy = bdx * cdy;
  cdxbdy = cdx * bdy;

  cdxady = cdx * ady;
  adxcdy = adx * cdy;

  adxbdy = adx * bdy;
  bdxady = bdx * ady;

  det = adz * (bdxcdy - cdxbdy) 
      + bdz * (cdxady - adxcdy)
      + cdz * (adxbdy - bdxady);

  permanent = (fabsf(bdxcdy) + fabsf(cdxbdy)) * fabsf(adz)
            + (fabsf(cdxady) + fabsf(adxcdy)) * fabsf(bdz)
            + (fabsf(adxbdy) + fabsf(bdxady)) * fabsf(cdz);
  errbound = o3derrboundA * permanent;
  if ((det > errbound) || (-det > errbound)) {
    return det;
  }

  return orient3dexact(pa, pb, pc, pd);
}




__host__ __device__ fp_tt static insphereexact(
        fp_tt const * const __restrict__ pa, 
        fp_tt const * const __restrict__ pb, 
        fp_tt const * const __restrict__ pc, 
        fp_tt const * const __restrict__ pd, 
        fp_tt const * const __restrict__ pe) {
    fp_tt axby1, bxcy1, cxdy1, dxey1, exay1;
    fp_tt bxay1, cxby1, dxcy1, exdy1, axey1;
    fp_tt axcy1, bxdy1, cxey1, dxay1, exby1;
    fp_tt cxay1, dxby1, excy1, axdy1, bxey1;
    fp_tt axby0, bxcy0, cxdy0, dxey0, exay0;
    fp_tt bxay0, cxby0, dxcy0, exdy0, axey0;
    fp_tt axcy0, bxdy0, cxey0, dxay0, exby0;
    fp_tt cxay0, dxby0, excy0, axdy0, bxey0;
    fp_tt ab[4], bc[4], cd[4], de[4], ea[4];
    fp_tt ac[4], bd[4], ce[4], da[4], eb[4];
    fp_tt temp8a[8], temp8b[8], temp16[16];
    int   temp8alen, temp8blen, temp16len;
    fp_tt abc[24], bcd[24], cde[24], dea[24], eab[24];
    fp_tt abd[24], bce[24], cda[24], deb[24], eac[24];
    int   abclen, bcdlen, cdelen, dealen, eablen;
    int   abdlen, bcelen, cdalen, deblen, eaclen;
    fp_tt temp48a[48], temp48b[48];
    int   temp48alen, temp48blen;
    fp_tt abcd[96], bcde[96], cdea[96], deab[96], eabc[96];
    int   abcdlen, bcdelen, cdealen, deablen, eabclen;
    fp_tt temp192[192];
    fp_tt det384x[384], det384y[384], det384z[384];
    int   xlen, ylen, zlen;
    fp_tt detxy[768];
    int   xylen;
    fp_tt adet[1152], bdet[1152], cdet[1152], ddet[1152], edet[1152];
    int   alen, blen, clen, dlen, elen;
    fp_tt abdet[2304], cddet[2304], cdedet[3456];
    int   ablen, cdlen;
    fp_tt deter[5760];
    int   deterlen;
    int   i;

    // Used by the macros to compute the approximate operation and error
    fp_tt bvirt;
    fp_tt avirt, bround, around;
    fp_tt c;
    fp_tt abig;
    fp_tt ahi, alo, bhi, blo;
    fp_tt err1, err2, err3;
    fp_tt _i, _j;
    fp_tt _0;

    // minor(a, b)
    Two_Product(pa[0], pb[1], axby1, axby0);  // ax*by
    Two_Product(pb[0], pa[1], bxay1, bxay0);  // bx*ay
    Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]); // ab = ax*by - bx*ay

    // minor(b, c)
    Two_Product(pb[0], pc[1], bxcy1, bxcy0);  // bx*cy
    Two_Product(pc[0], pb[1], cxby1, cxby0);  // cx*by
    Two_Two_Diff(bxcy1, bxcy0, cxby1, cxby0, bc[3], bc[2], bc[1], bc[0]); // bc = bx*cy - cx*by

    // minor(c, d)
    Two_Product(pc[0], pd[1], cxdy1, cxdy0);  // cx*dy
    Two_Product(pd[0], pc[1], dxcy1, dxcy0);  // dx*cy
    Two_Two_Diff(cxdy1, cxdy0, dxcy1, dxcy0, cd[3], cd[2], cd[1], cd[0]); // cd = cx*dy - dx*cy

    // minor(d, e)
    Two_Product(pd[0], pe[1], dxey1, dxey0);  // dx*ey
    Two_Product(pe[0], pd[1], exdy1, exdy0);  // ex*dy
    Two_Two_Diff(dxey1, dxey0, exdy1, exdy0, de[3], de[2], de[1], de[0]); // de = dx*ey - ex*dy

    // minor(e, a) = - minor(a, e)
    Two_Product(pe[0], pa[1], exay1, exay0);  // ex*ay
    Two_Product(pa[0], pe[1], axey1, axey0);  // ax*ey
    Two_Two_Diff(exay1, exay0, axey1, axey0, ea[3], ea[2], ea[1], ea[0]); // ea = ex*ay - ax*ey

    // minor(a, c)
    Two_Product(pa[0], pc[1], axcy1, axcy0);  // ax*cy
    Two_Product(pc[0], pa[1], cxay1, cxay0);  // cx*ay
    Two_Two_Diff(axcy1, axcy0, cxay1, cxay0, ac[3], ac[2], ac[1], ac[0]); // ac = ax*cy - cx*ay

    // minor(b, d)
    Two_Product(pb[0], pd[1], bxdy1, bxdy0);  // bx*dy
    Two_Product(pd[0], pb[1], dxby1, dxby0);  // dx*by
    Two_Two_Diff(bxdy1, bxdy0, dxby1, dxby0, bd[3], bd[2], bd[1], bd[0]); // bd = bx*dy - dx*by

    // minor(c, e)
    Two_Product(pc[0], pe[1], cxey1, cxey0);  // cx*ey
    Two_Product(pe[0], pc[1], excy1, excy0);  // ex*cy
    Two_Two_Diff(cxey1, cxey0, excy1, excy0, ce[3], ce[2], ce[1], ce[0]); // ce = cx*ey - ex*cy

    // minor(d, a) = - minor(a, d)
    Two_Product(pd[0], pa[1], dxay1, dxay0);  // dx*ay
    Two_Product(pa[0], pd[1], axdy1, axdy0);  // ax*dy
    Two_Two_Diff(dxay1, dxay0, axdy1, axdy0, da[3], da[2], da[1], da[0]); // da = dx*ay - ax*dy

    // minor(e, b) = - minor(b, e)
    Two_Product(pe[0], pb[1], exby1, exby0);  // ex*by
    Two_Product(pb[0], pe[1], bxey1, bxey0);  // bx*ey
    Two_Two_Diff(exby1, exby0, bxey1, bxey0, eb[3], eb[2], eb[1], eb[0]); // eb = ex*by - bx*ey

    // minor(a, b, c)
    temp8alen = scale_expansion_zeroelim(4, bc, pa[2], temp8a);           //   az*bc
    temp8blen = scale_expansion_zeroelim(4, ac, -pb[2], temp8b);          // - bz*ac
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b, temp16); // az * bc - bz * ac
    temp8alen = scale_expansion_zeroelim(4, ab, pc[2], temp8a);           //   cz*ab
    abclen    = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16, abc); // abc = ax*bx - bz*ac + cz*ab

    // minor(b, c, d)
    temp8alen = scale_expansion_zeroelim(4, cd, pb[2], temp8a);           //   bz*cd
    temp8blen = scale_expansion_zeroelim(4, bd, -pc[2], temp8b);          // - cz*bd
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b, temp16); // bz*cd - cz*bd
    temp8alen = scale_expansion_zeroelim(4, bc, pd[2], temp8a);           //   dz*bd
    bcdlen    = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16, bcd); // bcd = bz*cd - cz*bd + dz*bd

    // minor(c, d, e)
    temp8alen = scale_expansion_zeroelim(4, de, pc[2], temp8a);           //   cz*de
    temp8blen = scale_expansion_zeroelim(4, ce, -pd[2], temp8b);          // - dz*ce
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b, temp16); // cz*de - dz*ce
    temp8alen = scale_expansion_zeroelim(4, cd, pe[2], temp8a);           //   ez*cd
    cdelen    = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16, cde); // cde = cz*de - dz*ce + ez*cd

    // minor(d, e, a) = minor(a, d, e)
    temp8alen = scale_expansion_zeroelim(4, ea, pd[2], temp8a);           //   dz*ea
    temp8blen = scale_expansion_zeroelim(4, da, -pe[2], temp8b);          // - ez*da
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b, temp16); // dz*ea - ez*da
    temp8alen = scale_expansion_zeroelim(4, de, pa[2], temp8a);           //   az*de
    dealen    = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16, dea); // dea = dz*ea - ez*da + az*de

    // minor(e, a, b) = minor(a, b, e)
    temp8alen = scale_expansion_zeroelim(4, ab, pe[2], temp8a);           //   ez*ab
    temp8blen = scale_expansion_zeroelim(4, eb, -pa[2], temp8b);          // - az*eb
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b, temp16); // ez*ab - az*eb
    temp8alen = scale_expansion_zeroelim(4, ea, pb[2], temp8a);           //   bz*ea
    eablen    = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16, eab);    // eab = ez*ab - az*eb + bz*ea

    // minor(a, b, d)
    temp8alen = scale_expansion_zeroelim(4, bd, pa[2], temp8a);           //   az*bd
    temp8blen = scale_expansion_zeroelim(4, da, pb[2], temp8b);           //   bz*da
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b, temp16); // az*bd + bz*da
    temp8alen = scale_expansion_zeroelim(4, ab, pd[2], temp8a);           //   dz*ab
    abdlen    = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16, abd); // abd = az*bd + bz*da + dz*ab

    // minor(b, c, e) = minor(e, b, c)
    temp8alen = scale_expansion_zeroelim(4, ce, pb[2], temp8a);           //   bz*ce
    temp8blen = scale_expansion_zeroelim(4, eb, pc[2], temp8b);           //   cz*eb
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b, temp16); // bz*ce + cz*eb
    temp8alen = scale_expansion_zeroelim(4, bc, pe[2], temp8a);           //   ez*bc
    bcelen    = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16, bce); // bce = bz*ce + cz*eb + ez*bc

    // minor(c, d, a) = minor(a, c, d)
    temp8alen = scale_expansion_zeroelim(4, da, pc[2], temp8a);           //   cz*da
    temp8blen = scale_expansion_zeroelim(4, ac, pd[2], temp8b);           //   dz*ac
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b, temp16); // cz*da + dz*ac
    temp8alen = scale_expansion_zeroelim(4, cd, pa[2], temp8a);           //   az*cd
    cdalen    = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16, cda); // cda = cz*da + dz*ac + az*cd

    // minor(d, e, b) = minor(b, d, e)
    temp8alen = scale_expansion_zeroelim(4, eb, pd[2], temp8a);           //   dz*eb
    temp8blen = scale_expansion_zeroelim(4, bd, pe[2], temp8b);           //   ez*bd
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b, temp16); // dz*eb + ez*bd
    temp8alen = scale_expansion_zeroelim(4, de, pb[2], temp8a);           //   bz*de
    deblen    = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16, deb); // deb = dz*eb + ez*bd + bz*de

    // minor(e, a, c) = minor(a, c, e)
    temp8alen = scale_expansion_zeroelim(4, ac, pe[2], temp8a);           //   ez*ac
    temp8blen = scale_expansion_zeroelim(4, ce, pa[2], temp8b);           //   az*ce
    temp16len = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp8blen, temp8b, temp16); // ez*ac + az*ce
    temp8alen = scale_expansion_zeroelim(4, ea, pc[2], temp8a);           //   cz*ea
    eaclen    = fast_expansion_sum_zeroelim(temp8alen, temp8a, temp16len, temp16, eac); // eac = ez*ac + az*ce + cz*ea

    // minor(b, c, d, e)
    temp48alen = fast_expansion_sum_zeroelim(cdelen, cde, bcelen, bce, temp48a); // cde + bce
    temp48blen = fast_expansion_sum_zeroelim(deblen, deb, bcdlen, bcd, temp48b); // deb + bcd
    for (i = 0; i < temp48blen; i++) {
        temp48b[i] = -temp48b[i];
    }
    bcdelen = fast_expansion_sum_zeroelim(temp48alen, temp48a, temp48blen, temp48b, bcde); // bcde = cde + bcde - (deb + bcd)

    // minor(b, c, d, e)*alift
    xlen    = scale_expansion_zeroelim(bcdelen, bcde, pa[0], temp192); // bcde*ax
    xlen    = scale_expansion_zeroelim(xlen, temp192, pa[0], det384x); // bcde*ax*ax
    ylen    = scale_expansion_zeroelim(bcdelen, bcde, pa[1], temp192); // bcde*ay
    ylen    = scale_expansion_zeroelim(ylen, temp192, pa[1], det384y); // bcde*ay*ay
    zlen    = scale_expansion_zeroelim(bcdelen, bcde, pa[2], temp192); // bcde*az
    zlen    = scale_expansion_zeroelim(zlen, temp192, pa[2], det384z); // bcde*az*az
    xylen   = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy); // bcde*(ax*ax + ay*ay)
    alen    = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, adet);   // bcde*(ax*ax + ay*ay + az*az)

    // minor(c, d, e, a)
    temp48alen = fast_expansion_sum_zeroelim(dealen, dea, cdalen, cda, temp48a); // dea + cda
    temp48blen = fast_expansion_sum_zeroelim(eaclen, eac, cdelen, cde, temp48b); // eac + cde
    for (i = 0; i < temp48blen; i++) {
        temp48b[i] = -temp48b[i];
    }
    cdealen = fast_expansion_sum_zeroelim(temp48alen, temp48a, temp48blen, temp48b, cdea); // cdea = dea + cda - (eac + cde)
    
    // minor(c, d, e, a) * blift
    xlen    = scale_expansion_zeroelim(cdealen, cdea, pb[0], temp192); // cdea*bx
    xlen    = scale_expansion_zeroelim(xlen, temp192, pb[0], det384x); // cdea*bx*bx
    ylen    = scale_expansion_zeroelim(cdealen, cdea, pb[1], temp192); // cdea*by
    ylen    = scale_expansion_zeroelim(ylen, temp192, pb[1], det384y); // cdea*by*by
    zlen    = scale_expansion_zeroelim(cdealen, cdea, pb[2], temp192); // cdea*bz
    zlen    = scale_expansion_zeroelim(zlen, temp192, pb[2], det384z); // cdea*bz*bz
    xylen   = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy); // cdea*(bx*bx + by*by)
    blen    = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, bdet);   // cdea*(bx*bx + by*by + bz*bz)

    // minor(d, e, a, b)
    temp48alen = fast_expansion_sum_zeroelim(eablen, eab, deblen, deb, temp48a); // eab + deb
    temp48blen = fast_expansion_sum_zeroelim(abdlen, abd, dealen, dea, temp48b); // abd + dea
    for (i = 0; i < temp48blen; i++) {
        temp48b[i] = -temp48b[i];
    }
    deablen = fast_expansion_sum_zeroelim(temp48alen, temp48a, temp48blen, temp48b, deab); // deab = eab + deb - (abd + dea)

    // minor(d, e, a, b)*clift
    xlen    = scale_expansion_zeroelim(deablen, deab, pc[0], temp192); // deab*cx
    xlen    = scale_expansion_zeroelim(xlen, temp192, pc[0], det384x); // deab*cx*cx
    ylen    = scale_expansion_zeroelim(deablen, deab, pc[1], temp192); // deab*cy
    ylen    = scale_expansion_zeroelim(ylen, temp192, pc[1], det384y); // deab*cy*cy
    zlen    = scale_expansion_zeroelim(deablen, deab, pc[2], temp192); // deab*cz
    zlen    = scale_expansion_zeroelim(zlen, temp192, pc[2], det384z); // deab*cz*cz
    xylen   = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy); // deab*(cx*cx + cy*cy)
    clen    = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, cdet);   // deab*(cx*cx + cy*cy + cz*cz)

    // minor(e, a, b, c)
    temp48alen = fast_expansion_sum_zeroelim(abclen, abc, eaclen, eac, temp48a); // abc + eac
    temp48blen = fast_expansion_sum_zeroelim(bcelen, bce, eablen, eab, temp48b); // bce + eab
    for (i = 0; i < temp48blen; i++) {
        temp48b[i] = -temp48b[i];
    }
    eabclen = fast_expansion_sum_zeroelim(temp48alen, temp48a, temp48blen, temp48b, eabc); // eabc = abc + eac - (bce + eab)

    // minor(e, a, b, c)*dlift
    xlen    = scale_expansion_zeroelim(eabclen, eabc, pd[0], temp192);  // eabc*dx
    xlen    = scale_expansion_zeroelim(xlen, temp192, pd[0], det384x);  // eabc*dx*dx
    ylen    = scale_expansion_zeroelim(eabclen, eabc, pd[1], temp192);  // eabc*dy
    ylen    = scale_expansion_zeroelim(ylen, temp192, pd[1], det384y);  // eabc*dy*dy
    zlen    = scale_expansion_zeroelim(eabclen, eabc, pd[2], temp192);  // eabc*dz
    zlen    = scale_expansion_zeroelim(zlen, temp192, pd[2], det384z);  // eabc*dz*dz
    xylen   = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy); // eabc*(dx*dx + dy*dy)
    dlen    = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, ddet);   // eabc*(dx*dx + dy*dy + dz*dz)

    // minor(a, b, c, d)
    temp48alen = fast_expansion_sum_zeroelim(bcdlen, bcd, abdlen, abd, temp48a); // bcd + abd
    temp48blen = fast_expansion_sum_zeroelim(cdalen, cda, abclen, abc, temp48b); // cda + abc
    for (i = 0; i < temp48blen; i++) {
        temp48b[i] = -temp48b[i];
    }
    abcdlen = fast_expansion_sum_zeroelim(temp48alen, temp48a, temp48blen, temp48b, abcd); // abcd = bcd + abd - (cda + abc)

    // minor(a, b, c, d)*elift
    xlen    = scale_expansion_zeroelim(abcdlen, abcd, pe[0], temp192);  // abcd*ex
    xlen    = scale_expansion_zeroelim(xlen, temp192, pe[0], det384x);  // abcd*ex*ex
    ylen    = scale_expansion_zeroelim(abcdlen, abcd, pe[1], temp192);  // abcd*ey
    ylen    = scale_expansion_zeroelim(ylen, temp192, pe[1], det384y);  // abcd*ey*ey
    zlen    = scale_expansion_zeroelim(abcdlen, abcd, pe[2], temp192);  // abcd*ez
    zlen    = scale_expansion_zeroelim(zlen, temp192, pe[2], det384z);  // abcd*ez*ez
    xylen   = fast_expansion_sum_zeroelim(xlen, det384x, ylen, det384y, detxy); // abcd*(ex*ex + ey*ey)
    elen    = fast_expansion_sum_zeroelim(xylen, detxy, zlen, det384z, edet);   // abcd*(ex*ex + ey*ey + ez*ez)

    ablen    = fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet); // adet + bdet
    cdlen    = fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet); // cdet + ddet
    cdelen   = fast_expansion_sum_zeroelim(cdlen, cddet, elen, edet, cdedet); // cdet + ddet + edet
    deterlen = fast_expansion_sum_zeroelim(ablen, abdet, cdelen, cdedet, deter); // adet + bdet + cdet + ddet + edet

    return deter[deterlen - 1];
}

__host__ __device__ fp_tt static insphere(
        fp_tt const * const __restrict__ pa,
        fp_tt const * const __restrict__ pb,
        fp_tt const * const __restrict__ pc,
        fp_tt const * const __restrict__ pd,
        fp_tt const * const __restrict__ pe,
        fp_tt       * const __restrict__ err) {
    fp_tt aex, bex, cex, dex;
    fp_tt aey, bey, cey, dey;
    fp_tt aez, bez, cez, dez;
    fp_tt aexbey, bexaey, bexcey, cexbey, cexdey, dexcey, dexaey, aexdey;
    fp_tt aexcey, cexaey, bexdey, dexbey;
    fp_tt alift, blift, clift, dlift;
    fp_tt ab, bc, cd, da, ac, bd;
    fp_tt abc, bcd, cda, dab;
    fp_tt aezplus, bezplus, cezplus, dezplus;
    fp_tt aexbeyplus, bexaeyplus, bexceyplus, cexbeyplus;
    fp_tt cexdeyplus, dexceyplus, dexaeyplus, aexdeyplus;
    fp_tt aexceyplus, cexaeyplus, bexdeyplus, dexbeyplus;
    fp_tt det;
    fp_tt permanent;

    aex = pa[0] - pe[0];
    bex = pb[0] - pe[0];
    cex = pc[0] - pe[0];
    dex = pd[0] - pe[0];
    aey = pa[1] - pe[1];
    bey = pb[1] - pe[1];
    cey = pc[1] - pe[1];
    dey = pd[1] - pe[1];
    aez = pa[2] - pe[2];
    bez = pb[2] - pe[2];
    cez = pc[2] - pe[2];
    dez = pd[2] - pe[2];

    aexbey = aex * bey;
    bexaey = bex * aey;
    ab     = aexbey - bexaey;
    bexcey = bex * cey;
    cexbey = cex * bey;
    bc     = bexcey - cexbey;
    cexdey = cex * dey;
    dexcey = dex * cey;
    cd     = cexdey - dexcey;
    dexaey = dex * aey;
    aexdey = aex * dey;
    da     = dexaey - aexdey;

    aexcey = aex * cey;
    cexaey = cex * aey;
    ac     = aexcey - cexaey;
    bexdey = bex * dey;
    dexbey = dex * bey;
    bd     = bexdey - dexbey;

    abc = fmaf(aez, bc, fmaf(-bez, ac, cez * ab));
    bcd = fmaf(bez, cd, fmaf(-cez, bd, dez * bc));
    cda = fmaf(cez, da, fmaf(dez, ac, aez * cd));
    dab = fmaf(dez, ab, fmaf(aez, bd, bez * da));

    alift = fmaf(aex, aex, fmaf(aey, aey, aez * aez));
    blift = fmaf(bex, bex, fmaf(bey, bey, bez * bez));
    clift = fmaf(cex, cex, fmaf(cey, cey, cez * cez));
    dlift = fmaf(dex, dex, fmaf(dey, dey, dez * dez));

    det = fmaf(dlift, abc, fmaf(-clift,dab, fmaf(blift,cda, -alift*bcd)));

    aezplus    = fabsf(aez);
    bezplus    = fabsf(bez);
    cezplus    = fabsf(cez);
    dezplus    = fabsf(dez);
    aexbeyplus = fabsf(aexbey);
    bexaeyplus = fabsf(bexaey);
    bexceyplus = fabsf(bexcey);
    cexbeyplus = fabsf(cexbey);
    cexdeyplus = fabsf(cexdey);
    dexceyplus = fabsf(dexcey);
    dexaeyplus = fabsf(dexaey);
    aexdeyplus = fabsf(aexdey);
    aexceyplus = fabsf(aexcey);
    cexaeyplus = fabsf(cexaey);
    bexdeyplus = fabsf(bexdey);
    dexbeyplus = fabsf(dexbey);
    permanent  = fmaf(alift, 
                    fmaf(bezplus, 
                        (cexdeyplus + dexceyplus),
                        fmaf(cezplus, 
                            (dexbeyplus + bexdeyplus), 
                            dezplus * (bexceyplus + cexbeyplus))), 
                    fmaf(blift, 
                        fmaf(cezplus, 
                              (dexaeyplus + aexdeyplus),
                              fmaf(dezplus, 
                                  (aexceyplus + cexaeyplus), 
                                  aezplus * (cexdeyplus + dexceyplus))), 
                        fmaf(clift, 
                            fmaf(dezplus, 
                                (aexbeyplus + bexaeyplus), 
                                fmaf(aezplus, 
                                    (bexdeyplus + dexbeyplus), 
                                    bezplus * (dexaeyplus + aexdeyplus))), 
                            dlift * (
                            fmaf(aezplus,
                                (bexceyplus + cexbeyplus),
                                fmaf(bezplus, 
                                    (cexaeyplus + aexceyplus), 
                                    cezplus * (aexbeyplus + bexaeyplus))
                                )
                            )
                        )
                    )
                );
    *err = isperrboundA * permanent;
    return det;
}

#endif  // __STREAM_PREDICATES_HPP__
