#ifndef RESTRICTEDFOURBODY_H
#define RESTRICTEDFOURBODY_H

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <omp.h>
#include "capd/capdlib.h"


using namespace capd;
using namespace std;
using capd::autodiff::Node;

const double minDiamSize = 0.0012207031;
const double minSubdivision = 0.00012207031; // 1/8192
const double minSign = 0.00012207031;
const double minSizeBox = 0.1;
const double minNewton = 0.0001;
const double leftPerturbation = 0.0001;
const double rightPerturbation = 0.00011;

const interval withZero(-1.,1.);
const interval increase(-0.015, 0.015);


//Gradient of the potential (multiplied by (abc)^3 ) and its derivative
void gradPotential(Node t, Node in[], int dimIn, Node out[], int dimOut, Node param[], int noParams);
void derivativeGradPotential(Node t, Node in[], int dimIn, Node out[], int dimOut, Node param[], int noParams);
void derivativeGradPotentialExp(Node t, Node in[], int dimIn, Node out[], int dimOut, Node param[], int noParams);

//small functions
double maxi(double a, double b);
double mini(double a, double b);
double diamVect(const IVector & v);
bool isIncludedVect(const IVector & I1, const IVector & I2);
bool containsZeroVect(const IVector & I1);
interval absoluteValue(interval I);
IVector unionBox(const IVector & box1, const IVector & box2);


//functions to split
void splitBoxInStack(const IVector &initBox, stack<IVector> & res);
void splitHorizontalInStack(const IVector &initBox, stack<IVector> & res);
void splitVerticalInStack(const IVector &initBox, stack<IVector> & res);

//functions to verify if g' or g'' or g''' is non zero on a box
//(g obtained implicitely : f1(x,y) = 0 -> x = x(y)
// and then g(y)=f2(x(y),y) (up to permutation of f1/f2 and x/y) )
void computeJacobian(IMap &gradPot, IMap &deriv, const IVector &box, interval & res);
bool nonZeroJacobian(IMap &gradPot, IMap &deriv, const IVector &box);
bool nonZeroJacobianRecursive(IMap &gradPot, IMap &deriv, const IVector &box);
bool nonZeroGradPot(IMap &gradPot, const IVector &box);
void computeGSecond(IMap &gradPot, IMap &deriv, const IVector &box, const int & i, const int & j, interval & res);
bool nonZeroGSecond(IMap &gradPot, IMap &deriv, const IVector &box, const int & i, const int & j);
void computeGThird(IMap &gradPot, IMap &deriv, const IVector &box, const int & i, const int & j, interval & gThird);
bool nonZeroGThird(IMap &gradPot, IMap &deriv, const IVector &box, const int & i, const int & j);

//usefull and not too big functions
bool existijBox(IMap &gradPot, IMap &deriv, const IVector &box, int &ei, int &ej);
bool strongExistijBox(IMap &gradPot, IMap &deriv, const IVector &box, int & ei);
int signLine(IMap &gradPot, const IVector &initLine, int i, bool vertical);
bool nonZeroBorder(IMap &gradPot, IMap &deriv, const IVector &box, const int &i, const int &j);
void createGoodRectangle(const IVector &box, double & slope, vector<IVector> &rects);
bool isNewAtMostX(IMap &gradPot, IMap &deriv, IMap &derivExp, vector<IVector> & atMostX, const IVector &box, int X);

//bound on the number of zeros in a box
bool atMostOneZero(IMap &gradPot, IMap &deriv, const IVector &box);
bool atMostTwoZero(IMap &gradPot, IMap &deriv, const IVector &box);
bool atMostThreeZero(IMap &gradPot, IMap &deriv, IMap &derivExp, const IVector &box);

//big function
void noZero(const IVector &initBox, IMap &gradPot, IMap &deriv, IMap &derivExp, bool & lostInfo,  vector<IVector> & atMost1, vector<IVector> & atMost2, vector<IVector> & atMost3);

//removes boxes that are not needed
void removeAtMost1Box(IMap &gradPot, IMap &deriv, vector<IVector> & res);
void removeAtMostX(IMap &gradPot, IMap &deriv, IMap &derivExp, vector<IVector> & res, vector<IVector> & resY,  int X);
void removeInterAtMostXY(IMap &gradPot, IMap &deriv, IMap &derivExp, vector<IVector> & atMostX, vector<IVector> & atMostY, int X, int Y);
void replace12by3(IMap &gradPot, IMap &deriv, IMap &derivExp, vector<IVector> & atMost1, vector<IVector> & atMost2, vector<IVector> & atMost3);
void bigBoxAtMost3(IMap &gradPot, IMap &deriv, IMap &derivExp, vector<IVector> & atMost3);

//Newton related functions (transform atMost1 boxes into box with exactly 1 zero)
IVector computeN(const IVector &box, IMap &gradPot, IMap &deriv);
IVector computeK(const IVector &box, IMap &gradPot, IMap &deriv);
int NewtonIteration(const IVector &box, IMap &gradPot, IMap &deriv, IVector & res, bool withN);
int recursiveNewton(const IVector &box, IMap &gradPot, IMap &deriv, IVector & res, bool withN);
void transformAtMost1IntoGoodBoxes(IMap &gradPot, IMap &deriv, vector<IVector> &posOfSol, vector<IVector> &atMost1);

//the total function
int totalNoZero(const IVector &box, IMap &gradPot, IMap &deriv, IMap &derivExp, bool & lostInfo, vector<IVector> & res, vector<IVector> & atMost1, vector<IVector> & atMost2, vector<IVector> & atMost3);

int createFunctionAndZeros(const interval &m1, const interval & m2, const interval & m3, const IVector &box, vector<IVector> & posOfZero, bool & lostInfo, vector<IVector> & atMost1, vector<IVector> & atMost2, vector<IVector> & atMost3);

//test function
void testDifferentMasses(ofstream & notGood, ofstream & inf11sol, ofstream & e11sol, ofstream & e12sol, ofstream & e13sol, ofstream &more13);

#endif

