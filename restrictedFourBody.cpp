#include "restrictedFourBody.hpp"


/////////////************** Gradient of the potential and its derivative ************/////////////////

//Function computing the gradient of the potential (the function we want to find the zeros)
//To avoid denominator, the gradient is multiplied by a³b³c³, so it is not exactly the gradient
void gradPotential(Node t, Node in[], int dimIn, Node out[], int dimOut, Node param[], int noParams)
{

	Node sqrt3 = param[5]; //sqrt(3) is not representable so given as a param
	Node oneHalf = param[6];

	Node ax = in[0]-sqrt3/3.; //x-coordinate of vector (m1,x)
	Node ay = in[1];
	Node bx = in[0]+sqrt3/6.;
	Node by = in[1]-oneHalf;
	Node cx = in[0]+sqrt3/6.;
	Node cy = in[1]+oneHalf;

	Node axSquare = ax^2;
	Node aySquare = ay^2;
	Node bxSquare = bx^2;
	Node bySquare = by^2;
	Node cxSquare = cx^2;
	Node cySquare = cy^2;
	
	Node a = sqrt(axSquare + aySquare); //distance between m1 and x
	Node b = sqrt(bxSquare + bySquare);
	Node c = sqrt(cxSquare + cySquare);
	
	Node factor1 = param[2]*(b^3)*(c^3); // cx and cy are params 0 and 1, so mi are param[i+1]
	Node factor2 = param[3]*(a^3)*(c^3);
	Node factor3 = param[4]*(a^3)*(b^3);
	
	out[0] = (a^3)*(b^3)*(c^3)*(in[0]-param[0]) -factor1*ax-factor2*bx-factor3*cx;
	out[1] = (a^3)*(b^3)*(c^3)*(in[1]-param[1]) -factor1*ay-factor2*by-factor3*cy;
}


//Function computing the Jacobian matrix of gradPotential
//the first 2 output are df/dx and the last 2 are df/dy
void derivativeGradPotential(Node t, Node in[], int dimIn, Node out[], int dimOut, Node param[], int noParams)
{

	Node sqrt3 = param[5]; //sqrt(3) is not representable so given as a param
	Node oneHalf = param[6];

	Node ax = in[0]-sqrt3/3.; //x-coordinate of vector (m1,x)
	Node ay = in[1];
	Node bx = in[0]+sqrt3/6.;
	Node by = in[1]-oneHalf;
	Node cx = in[0]+sqrt3/6.;
	Node cy = in[1]+oneHalf;
	
	Node a = sqrt((ax^2) + (ay^2)); //distance between m1 and x
	Node b = sqrt((bx^2) + (by^2));
	Node c = sqrt((cx^2) + (cy^2));

	Node factor1x = 3*a*b*c*(ax*(b^2)*(c^2) + (a^2)*bx*(c^2) + (a^2)*(b^2)*cx);
	Node factor1y = 3*a*b*c*(ay*(b^2)*(c^2) + (a^2)*by*(c^2) + (a^2)*(b^2)*cy);
	Node factor2x = 3*b*c*(bx*(c^2)+(b^2)*cx);
	Node factor2y = 3*b*c*(by*(c^2)+(b^2)*cy);
	Node factor3x = 3*a*c*(ax*(c^2)+(a^2)*cx);
	Node factor3y = 3*a*c*(ay*(c^2)+(a^2)*cy);
	Node factor4x = 3*a*b*(ax*(b^2)+(a^2)*bx);
	Node factor4y = 3*a*b*(ay*(b^2)+(a^2)*by);

	// df1/dx
	out[0] = (a^3)*(b^3)*(c^3)+(in[0]-param[0])*factor1x - param[2]*((b^3)*(c^3)+ax*factor2x) - param[3]*((a^3)*(c^3)+bx*factor3x) - param[4]*((a^3)*(b^3)+cx*factor4x);

	// df2/dx
	out[1] = (in[1]-param[1])*factor1x - param[2]*ay*factor2x - param[3]*by*factor3x - param[4]*cy*factor4x;

	// df1/dy
	out[2] = (in[0]-param[0])*factor1y - param[2]*ax*factor2y - param[3]*bx*factor3y - param[4]*cx*factor4y;

	// df2/dy
	out[3] = (a^3)*(b^3)*(c^3)+(in[1]-param[1])*factor1y - param[2]*((b^3)*(c^3)+ay*factor2y) - param[3]*((a^3)*(c^3)+by*factor3y) - param[4]*((a^3)*(b^3)+cy*factor4y);

}

//Same function as the previous one but with exp instead of power 3 
//(automatic differentiation of order >= 2 is not implemented for power 3)
void derivativeGradPotentialExp(Node t, Node in[], int dimIn, Node out[], int dimOut, Node param[], int noParams)
{

	Node sqrt3 = param[5]; //sqrt(3) is not representable so given as a param
	Node oneHalf = param[6];

	Node ax = in[0]-sqrt3/3.; //x-coordinate of vector (m1,x)
	Node ay = in[1];
	Node bx = in[0]+sqrt3/6.;
	Node by = in[1]-oneHalf;
	Node cx = in[0]+sqrt3/6.;
	Node cy = in[1]+oneHalf;
	
	Node a = sqrt((ax^2) + (ay^2)); //distance between m1 and x
	Node b = sqrt((bx^2) + (by^2));
	Node c = sqrt((cx^2) + (cy^2));

	Node factor1x = 3*a*b*c*(ax*(b^2)*(c^2) + (a^2)*bx*(c^2) + (a^2)*(b^2)*cx);
	Node factor1y = 3*a*b*c*(ay*(b^2)*(c^2) + (a^2)*by*(c^2) + (a^2)*(b^2)*cy);
	Node factor2x = 3*b*c*(bx*(c^2)+(b^2)*cx);
	Node factor2y = 3*b*c*(by*(c^2)+(b^2)*cy);
	Node factor3x = 3*a*c*(ax*(c^2)+(a^2)*cx);
	Node factor3y = 3*a*c*(ay*(c^2)+(a^2)*cy);
	Node factor4x = 3*a*b*(ax*(b^2)+(a^2)*bx);
	Node factor4y = 3*a*b*(ay*(b^2)+(a^2)*by);

	// df1/dx
	out[0] = exp(3.*log(a*b*c))+(in[0]-param[0])*factor1x - param[2]*(exp(3.*log(b*c))+ax*factor2x) - param[3]*(exp(3.*log(a*c))+bx*factor3x) - param[4]*(exp(3.*log(a*b))+cx*factor4x);

	// df2/dx
	out[1] = (in[1]-param[1])*factor1x - param[2]*ay*factor2x - param[3]*by*factor3x - param[4]*cy*factor4x;

	// df1/dy
	out[2] = (in[0]-param[0])*factor1y - param[2]*ax*factor2y - param[3]*bx*factor3y - param[4]*cx*factor4y;

	// df2/dy
	out[3] = exp(3.*log(a*b*c)) +(in[1]-param[1])*factor1y - param[2]*(exp(3.*log(b*c))+ay*factor2y) - param[3]*(exp(3.*log(a*c))+by*factor3y) - param[4]*(exp(3.*log(a*b))+cy*factor4y);

}


///////////************* small functions **************///////////////////

double maxi(double a, double b)
{
	if (a > b)
		return a;
	return b;
}

double mini(double a, double b)
{
	if (a < b)
		return a;
	return b;
}


double diamVect(const IVector & v)
{
	return diam(v[0]).rightBound()+diam(v[1]).rightBound();
}


bool isIncludedVect(const IVector & I1, const IVector & I2)
{
	if (I2[0].contains(I1[0]) && I2[1].contains(I1[1]) )
		return true;
	else
		return false;
}

bool containsZeroVect(const IVector & I1)
{
	if (I1[0].contains(0.) && I1[1].contains(0.) )
		return true;
	else
		return false;
}


interval absoluteValue(interval I)
{
	double Il = I.leftBound();
	double Ir = I.rightBound();
	double sup = maxi(abs(Il), abs(Ir) );
	double inf = 0;

	if (Il*Ir > 0)
		inf = mini(abs(Il), abs(Ir) );

	interval abso(inf, sup);
	return abso;
}


IVector unionBox(const IVector & box1, const IVector & box2)
{
	IVector unionB(2);
	unionB[0] = intervalHull(box1[0], box2[0]);
	unionB[1] = intervalHull(box1[1], box2[1]);
	return unionB;
}


////////////////////************ Split a box ***************//////////////////

//split a 2D box into 4 sub-boxes and put the subBoxes on the top of a stack
void splitBoxInStack(const IVector &initBox, stack<IVector> & res)
{
	interval I1 = initBox[0];
	interval I2 = initBox[1];

	double d0 = diam(initBox[0]).rightBound();
	double d1 = diam(initBox[1]).rightBound();

	//split the intervals into 2 overlapping intervals
	interval I1l(I1.leftBound(), mid(I1).rightBound()+rightPerturbation*d0);
	interval I1r(mid(I1).leftBound()-leftPerturbation*d0, I1.rightBound() );
	interval I2l(I2.leftBound(), mid(I2).rightBound()+rightPerturbation*d1);
	interval I2r(mid(I2).leftBound()-leftPerturbation*d1, I2.rightBound() );

	interval tmp[] = {I1l, I2l};
	res.push(IVector(2, tmp) );

	tmp[1] = I2r;
	res.push(IVector(2, tmp) );
	tmp[0] = I1r;
	res.push(IVector(2, tmp) );
	tmp[1] = I2l;
	res.push(IVector(2, tmp) );
}


//split a 2D box into 2 horizontal sub-boxes and put the result on the top of a stack
void splitHorizontalInStack(const IVector &initBox, stack<IVector> & res)
{
	interval I2 = initBox[1];

	double d1 = diam(initBox[1]).rightBound();

	if (d1 > minDiamSize/10.)
	{
		//split the intervals into 2 overlapping intervals
		interval I2l(I2.leftBound(), mid(I2).rightBound()+rightPerturbation*d1);
		interval I2r(mid(I2).leftBound()-leftPerturbation*d1, I2.rightBound() );

		interval tmp[] = {initBox[0], I2l};
		res.push(IVector(2, tmp) );

		tmp[1] = I2r;
		res.push(IVector(2, tmp) );
	}
	else //if the box is too horizontal, we split it vertically
		splitVerticalInStack(initBox, res);
}


//split a 2D box into 2 vertical sub-boxes and put the result on the top of a stack
void splitVerticalInStack(const IVector &initBox, stack<IVector> & res)
{
	interval I1 = initBox[0];

	double d0 = diam(initBox[0]).rightBound();

	if (d0 > minDiamSize/10.)
	{
		//split the intervals into 2 overlapping intervals
		interval I1l(I1.leftBound(), mid(I1).rightBound()+rightPerturbation*d0);
		interval I1r(mid(I1).leftBound()-leftPerturbation*d0, I1.rightBound() );

		interval tmp[] = {I1l, initBox[1]};
		res.push(IVector(2, tmp) );

		tmp[0] = I1r;
		res.push(IVector(2, tmp) );
	}
	else //if the box is too vertical, we split it horizontally
		splitHorizontalInStack(initBox, res);
		
}


///////////////////******** NonZero Something **********/////////////////////


//compute the determinant of Df over box and put the result in res
void computeJacobian(IMap &gradPot, IMap &deriv, const IVector &box, interval & res)
{
	//IMatrix Df = gradPot.derivative(box);
	IMatrix Df(2,2);
	IVector tmp = deriv(box);
	Df[0][0] = tmp[0];
	Df[1][0] = tmp[1];
	Df[0][1] = tmp[2];
	Df[1][1] = tmp[3];

	res = Df[0][0]*Df[1][1]-Df[0][1]*Df[1][0];
	
}


//test if the determinant of the IMatrix Df(box) is non zero
bool nonZeroJacobian(IMap &gradPot, IMap &deriv, const IVector &box)
{
	interval jac;
	computeJacobian(gradPot, deriv, box, jac);

	if (!jac.contains(0.) )
		return true;
	else
		return false;
}


//return true if the jacobian is non zero on box, false if we cannot be sure that it is non zero
bool nonZeroJacobianRecursive(IMap &gradPot, IMap &deriv, const IVector &box)
{
	stack<IVector> waitingBoxes;
	waitingBoxes.push(box);

	while (!waitingBoxes.empty())
	{
		IVector topBox = waitingBoxes.top();
		waitingBoxes.pop();
		
		interval jac;
		computeJacobian(gradPot, deriv, topBox, jac);

		//if jac contains no zero, we just remove the box
		//else if the box is too small it's a failure
		//else we subdivide the box and put the small boxes in the stack
		if (jac.contains(0.))
		{
			if (diamVect(topBox) < minSubdivision)
			{
				return false;
			}
			else
				splitBoxInStack(topBox, waitingBoxes);
		}
				
	}

	return true;
}

//return true if the box contains no zero of gradPot
bool nonZeroGradPot(IMap &gradPot, const IVector &box)
{
	stack<IVector> waitingBoxes;
	waitingBoxes.push(box);

	while (!waitingBoxes.empty())
	{
		IVector topBox = waitingBoxes.top();
		waitingBoxes.pop();
		
		IVector image = gradPot(topBox);

		//if image contains no zero, we just remove the box
		//else if the box is too small it's a failure
		//else we subdivide the box and put the small boxes in the stack
		if (containsZeroVect(image))
		{
			if (diamVect(topBox) < minSubdivision)
				return false;
			else
				splitBoxInStack(topBox, waitingBoxes);
		}
				
	}

	return true;
}


//compute the value of g'' over box and put the result in gSecond, provided that we have 
//(i,j) such that dfi/dxj != 0, (i = 0 or 1, idem for j)
//The value is computed from deriv because else we would need to modify the definition of gradPot
//(because of the cubes)
void computeGSecond(IMap &gradPot, IMap &deriv, const IVector &box, const int & i, const int & j, interval & gSecond)
{
	IMatrix Df(2,2);
	IMatrix Hf(4,2);
	IVector tmp = deriv(box, Hf);
	Df[0][0] = tmp[0];
	Df[1][0] = tmp[1];
	Df[0][1] = tmp[2];
	Df[1][1] = tmp[3];

	if (Df[i][j].contains(0.))
	{
		#pragma omp critical (cerr)
			cerr << "attention, I cannot compute gSecond on this box, division by zero. I answer : " << withZero << endl;
		gSecond = withZero;
	}
	else
	{
		interval xjPrime = -Df[i][1-j]/Df[i][j];
		interval xjSecond = -(Hf[i+2*j][j]*power(xjPrime,2)
                              +(Hf[i+2*j][1-j]+Hf[i+2*(1-j)][j])*xjPrime
                              +Hf[i+2*(1-j)][1-j])
                             /Df[i][j];
		gSecond = Hf[1-i+2*j][j]*power(xjPrime,2)+(Hf[1-i+2*j][1-j]+Hf[1-i+2*(1-j)][j])*xjPrime+Hf[1-i+2*(1-j)][1-j]
		          +Df[1-i][j]*xjSecond;
	}
	
}

//Verify recursively that GSecond is non zero on box
//return true if GSecond is non zero on box, false if we are not able to prove so
bool nonZeroGSecond(IMap &gradPot, IMap &deriv, const IVector &box, const int & i, const int & j)
{
	stack<IVector> waitingBoxes;
	waitingBoxes.push(box);

	while (!waitingBoxes.empty())
	{
		IVector topBox = waitingBoxes.top();
		waitingBoxes.pop();
		
		interval gSecond;
		computeGSecond(gradPot, deriv, topBox, i,j, gSecond);

		if (gSecond.contains(0.))
		{
			if (diamVect(topBox) < minSubdivision)
				return false;
			else
				splitBoxInStack(topBox, waitingBoxes);
		}
				
	}

	return true;
}


//compute the value of g''' over box and put the result in gThird, provided that we have 
//(i,j) such that dfi/dxj != 0, (i = 0 or 1, idem for j)
void computeGThird(IMap &gradPot, IMap &deriv, const IVector &box, const int & i, const int & j, interval & gThird)
{
	IMatrix Df(2,2);
	IMatrix Hf(4,2);
	IHessian Hf3(4,2);
	IVector tmp = deriv(box, Hf, Hf3);
	Df[0][0] = tmp[0];
	Df[1][0] = tmp[1];
	Df[0][1] = tmp[2];
	Df[1][1] = tmp[3];


	if (Df[i][j].contains(0.))
	{
		#pragma omp critical (cerr)
			cerr << "attention, I cannot compute gThird on this box, division by zero" << endl;
		gThird = withZero;
	}
	else
	{

		interval xjPrime = -Df[i][1-j]/Df[i][j];

		interval xjSecond = -( Hf[i+2*j][j]*power(xjPrime,2)
		                      +(Hf[i+2*j][1-j]+Hf[i+2*(1-j)][j])*xjPrime
		                      +Hf[i+2*(1-j)][1-j])
		                    /Df[i][j];

		interval xThird = -( Hf3(i+2*j, j, j)*power(xjPrime, 3)
		                     +3.*Hf3(i+2*j, j, 1-j)*power(xjPrime, 2)
		                     +3.*Hf3(i+2*j, 1-j, 1-j)*xjPrime
		                     +Hf3(i+2*(1-j), 1-j, 1-j)
		                     +3.*Hf[i+2*j][j]*xjPrime*xjSecond
		                     +3.*Hf[i+2*j][1-j]*xjSecond)
		                   /Df[i][j];

		gThird = Hf3(1-i+2*j, j, j)*power(xjPrime, 3)
		         +3.*Hf3(1-i+2*j, j, 1-j)*power(xjPrime, 2)
		         +3.*Hf3(1-i+2*j, 1-j, 1-j)*xjPrime
		         +Hf3(1-i+2*(1-j), 1-j, 1-j)
		         +3.*Hf[1-i+2*j][j]*xjPrime*xjSecond
		         +3.*Hf[1-i+2*j][1-j]*xjSecond
		         +Df[1-i][j]*xThird;

	}
	
}

//Verify recursively that GThird is non zero on box
//return true if GThird is non zero on box, false if we are not able to prove so
bool nonZeroGThird(IMap &gradPot, IMap &deriv, const IVector &box, const int & i, const int & j)
{
	stack<IVector> waitingBoxes;
	waitingBoxes.push(box);

	while (!waitingBoxes.empty())
	{
		IVector topBox = waitingBoxes.top();
		waitingBoxes.pop();
		
		interval gThird;
		computeGThird(gradPot, deriv, topBox, i,j, gThird);

		if (gThird.contains(0.))
		{
			if (diamVect(topBox) < minSubdivision)
				return false;
			else
				splitBoxInStack(topBox, waitingBoxes);
		}
				
	}

	return true;
}


////////////************ Functions for verifying we can use implicit method *********//////////////

//return true if exist i j such that dfi/dxj is non zero over the entire box
//and put in ei and ej the value of i and j for which dfi/dxj is non zero
bool existijBox(IMap &gradPot, IMap &deriv, const IVector &box, int &ei, int &ej)
{

	//usually it is sufficient to look only at deriv(box) without recursion
	IVector D_f = deriv(box);
	if (!D_f[0].contains(0.) )
	{
		ei = 0;
		ej = 0;
		return true;
	}
	else if (!D_f[1].contains(0.) )
	{
		ei = 1;
		ej = 0;
		return true;
	}
	else if (!D_f[2].contains(0.) )
	{
		ei = 0;
		ej = 1;
		return true;
	}
	else if (!D_f[3].contains(0.) )
	{
		ei = 1;
		ej = 1;
		return true;
	}

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			stack<IVector> waitingBoxes;
			waitingBoxes.push(box);
			bool tooSmall = false;

			while (!waitingBoxes.empty() && !tooSmall)
			{
				IVector topBox = waitingBoxes.top();
				waitingBoxes.pop();
		
				IVector Df = deriv(topBox);

				//if Df[i][j] contains no zero, we just remove the box
				//else if the box is too small it's a failure, we try another (i,j)
				//else we subdivide the box and put the small boxes in the stack
				if (Df[i+2*j].contains(0.))
				{
					if (diamVect(topBox) < minSubdivision)
						tooSmall = true;
					else
						splitBoxInStack(topBox, waitingBoxes);
				}
				
			}

			if (waitingBoxes.empty()) //if we stop because no more box, we have found i,j such that dfi/dxj != 0
			{
				ei = i;
				ej = j;
				return true;
			}
		}
	}
	return false;
}

//return true if exist i such that dfi/dx and dfi/dy are non zero over the entire box
//(sufficient condition to use nonZero jacobian/Gsecond/Gthird on the box)
//put the good i in ei
bool strongExistijBox(IMap &gradPot, IMap &deriv, const IVector &box, int & ei)
{
	//usually it is sufficient to look only at deriv(box) without recursion
	IVector D_f = deriv(box);
//cout << "strongExistij on box " << box << ", Df = " << D_f << endl;

	if (!D_f[0].contains(0.) && !D_f[2].contains(0.))
	{
		ei = 0;
		return true;
	}
	else if (!D_f[1].contains(0.) && !D_f[3].contains(0.) )
	{
		ei = 1;
		return true;
	}

	for (int i = 0; i < 2; i++)
	{
		bool goodi = true;
		for (int j = 0; j < 2; j++)
		{
			stack<IVector> waitingBoxes;
			waitingBoxes.push(box);
			bool tooSmall = false;

			while (!waitingBoxes.empty() && !tooSmall && goodi)
			{
				IVector topBox = waitingBoxes.top();
				waitingBoxes.pop();
		
				IVector Df = deriv(topBox);

				//if Df[i][j] contains no zero, we just remove the box
				//else if the box is too small it's a failure, we try another (i,j)
				//else we subdivide the box and put the small boxes in the stack
				if (Df[i+2*j].contains(0.))
				{
					if (diamVect(topBox) < minSubdivision)
						tooSmall = true;
					else
						splitBoxInStack(topBox, waitingBoxes);
				}
				
			}

			if (!waitingBoxes.empty()) //if we stop because too small box, this is not a good i
			{
				goodi = false;
			}
		}
		if (goodi)
		{
			ei = i;
			return true;
		}

	}
	return false;
}


//vertical is true if the line is vertical, false if the line is horizontal
//return -1 if gradPot[i](line) < 0, +1 if gradPot[i](line) > 0, and 0 if don't know
int signLine(IMap &gradPot, const IVector &initLine, int i, bool vertical)
{
	stack<IVector> waitingLines;
	waitingLines.push(initLine);

	int sign = 0;

	while (!waitingLines.empty())
	{
		IVector line = waitingLines.top();
		waitingLines.pop();
		interval value = gradPot(line)[i];

		//if sign = 0 (initial config), we need to determine the sign
		//if we already have the sign, we look if it's the same
		//if we cannot say the sign and the box is too small, we return 0
		//else we subdivide the box and put the small boxes in the stack
		if (value < 0.)
		{
			if (sign == 0)
				sign = -1;
			else if (sign == 1)
				return 0;
		}
		else if (value > 0.)
		{
			if (sign == 0)
				sign = 1;
			else if (sign == -1)
				return 0;
		}
		else
		{
			if (diamVect(line) < minSign)
				return 0;
			else if (vertical)
				splitHorizontalInStack(line, waitingLines);
			else
				splitVerticalInStack(line, waitingLines);
		}
				
	}

	return sign;
}



//return true if fi keep the same sign on the sides defined by xj=xjmin and xj = xjmax
//(that is fi = 0 does not cross the sides xj=xjmin and xj = xjmax )
//return false otherwise
bool nonZeroBorder(IMap &gradPot, IMap &deriv, const IVector &box, const int &i, const int &j)
{
	int sign1 = 0, sign2 = 0;
	if (j == 0)
	{
		IVector leftSide = box;
		leftSide[0] = leftSide[0].leftBound();
		IVector rightSide = box;
		rightSide[0] = rightSide[0].rightBound();
		sign1 = signLine(gradPot, leftSide, i, true);
		sign2 = signLine(gradPot, rightSide, i, true);
	}
	else
	{
		IVector upSide = box;
		upSide[1] = upSide[1].rightBound();
		IVector downSide = box;
		downSide[1] = downSide[1].leftBound();
		sign1 = signLine(gradPot, downSide, i, false);
		sign2 = signLine(gradPot, upSide, i, false);
	}

	int sign = sign1*sign2;

	if (sign != 0)
		return true;

	return false;
}


///////////////************** at most X zeros in a box ***********////////////////

//the function return true if we can prove that there are at most 1 zero in box
//false otherwise
bool atMostOneZero(IMap &gradPot, IMap &deriv, const IVector &box)
{
	int i = -1, j = -1;
	if (strongExistijBox(gradPot, deriv, box, i))
		return nonZeroJacobianRecursive(gradPot, deriv, box);

	else
	{
		bool weakExist = existijBox(gradPot, deriv, box, i,j);
		if (weakExist && i != -1 && j != -1 && nonZeroBorder(gradPot, deriv, box, i, j) )
			return nonZeroJacobianRecursive(gradPot, deriv, box);

		else //try increase the size of the box
		{
			IVector bigBox = box;
			bigBox[0]+= increase;
			bigBox[1]+= increase;
			weakExist = existijBox(gradPot, deriv, bigBox, i,j);
			if (weakExist && i != -1 && j != -1 && nonZeroBorder(gradPot, deriv, bigBox, i, j) )
				return nonZeroJacobianRecursive(gradPot, deriv, bigBox);
		}
	}

	return false;

}

//the function return true if we can prove that there are at most 2 zeros in box
//false otherwise
bool atMostTwoZero(IMap &gradPot, IMap &deriv, const IVector &box)
{
	int i=-1;
	bool exist = strongExistijBox(gradPot, deriv, box, i);
	if (exist && i != -1 )
	{
		return (nonZeroGSecond(gradPot, deriv, box, i, 0) || nonZeroGSecond(gradPot, deriv, box, i, 1) );
	}

	else
	{
		i = -1;
		int j = -1;
		bool weakExist = existijBox(gradPot, deriv, box, i,j);
		if (weakExist && i != -1 && j != -1 && nonZeroBorder(gradPot, deriv, box, i, j) )
			return nonZeroGSecond(gradPot, deriv, box, i, j);

		else //try increase the size of the box
		{
			IVector bigBox = box;
			bigBox[0]+= increase;
			bigBox[1]+= increase;
			weakExist = existijBox(gradPot, deriv, bigBox, i,j);
			if (weakExist && i != -1 && j != -1 && nonZeroBorder(gradPot, deriv, bigBox, i, j) )
				return nonZeroGSecond(gradPot, deriv, bigBox, i, j);
		}
	}

	return false;
	
}

//the function return true if we can prove that there are at most 3 zeros in box
//false otherwise
bool atMostThreeZero(IMap &gradPot, IMap &deriv, IMap &derivExp, const IVector &box)
{
	int i=-1;
	bool exist = strongExistijBox(gradPot, deriv, box, i);
	if (exist && i != -1 )
	{
		return (nonZeroGThird(gradPot, derivExp, box, i, 0) || nonZeroGThird(gradPot, derivExp, box, i, 1) );
	}

	else
	{
		i = -1;
		int j = -1;
		bool weakExist = existijBox(gradPot, deriv, box, i,j);
		if (weakExist && i != -1 && j != -1 && nonZeroBorder(gradPot, deriv, box, i, j) )
			return nonZeroGThird(gradPot, derivExp, box, i, j);

		else //try increase the size of the box
		{
			IVector bigBox = box;
			bigBox[0]+= increase;
			bigBox[1]+= increase;
			weakExist = existijBox(gradPot, deriv, bigBox, i,j);
			if (weakExist && i != -1 && j != -1 && nonZeroBorder(gradPot, deriv, bigBox, i, j) )
				return nonZeroGThird(gradPot, derivExp, bigBox, i, j);
		}
	}

	return false;
	
}


//////////************* Bound the number of zeros in a box ************///////////////////

//X is for saying if we are dealing with atMost1, atMost2 or atMost3 boxes.
//Check if the new box we want to add to atMostX is usefull or not.
//Return false if we can find another box box2 in atMostX such that the union of box2 and box contains
//at most X zero(s). In that case, box2 is replaced by the union box in atMostX.
//Return true otherwise.
bool isNewAtMostX(IMap &gradPot, IMap &deriv, IMap &derivExp, vector<IVector> & atMostX, const IVector &box, int X)
{
	for (unsigned int i = 0; i < atMostX.size(); i++)
	{
		IVector tmp(2);
		bool intersect = intersection(atMostX[i], box, tmp);

		if (intersect)
		{	
			IVector unionB = unionBox(atMostX[i], box);
			try
			{
				if (X == 1 && atMostOneZero(gradPot, deriv, unionB) )
				{
					atMostX[i] = unionB;
					return false;
				}
				else if (X == 2 && atMostTwoZero(gradPot, deriv, unionB) )
				{
					atMostX[i] = unionB;
					return false;
				}
				else if (X == 3 && atMostThreeZero(gradPot, deriv, derivExp, unionB) )
				{
					atMostX[i] = unionB;
					return false;
				}
				else if (X != 1 && X != 2 && X != 3)
					cerr << "Attention, bad argument in isNewAtMostX" << endl;

			}
			catch (exception& e)
			{
				#pragma omp critical (cerr)
					cerr << "I avoid quitting because of : " << e.what() << endl;
			}
		}
	}
	return true;
}


//Return boxes where we are able to bound the number of zeros.
//Boxes are put in res if there are exactly one zero inside, in atMostX if there are at most X zeros inside.
//lostInfo is put to true if there are boxes that are too small to continue and where we are not able
//to bound the number of zeros.

void noZero(const IVector &initBox, IMap &gradPot, IMap &deriv, IMap &derivExp, bool & lostInfo,  vector<IVector> & atMost1, vector<IVector> & atMost2, vector<IVector> & atMost3)
{
	stack<IVector> waitingBoxes;
	waitingBoxes.push(initBox);

	while ( !waitingBoxes.empty() )
	{
		IVector box = waitingBoxes.top();
		waitingBoxes.pop();

		IVector image = gradPot(box);

		if (!image[0].contains(0.) || !image[1].contains(0.) )
		{
			;
		}

		else if (diamVect(box) > minDiamSize)
		{
			splitBoxInStack(box, waitingBoxes);
		}

		else if (nonZeroGradPot(gradPot, box))
		{
			;
		}

		else if (atMostOneZero(gradPot, deriv, box) )
		{
			//we add the box only if it is usefull (verification on the flight, and not at the end)
			if (isNewAtMostX(gradPot, deriv, derivExp, atMost1, box, 1) )
				atMost1.push_back(box);
		}

		else if (atMostTwoZero(gradPot, deriv, box) )
		{
			//add the box only if it is usefull
			if (isNewAtMostX(gradPot, deriv, derivExp, atMost2, box, 2) )
				atMost2.push_back(box);
		}

		else if (atMostThreeZero(gradPot, deriv, derivExp, box) )
		{
			//add the box only if it is usefull
			if (isNewAtMostX(gradPot, deriv, derivExp, atMost3, box, 3) )
				atMost3.push_back(box);
		}

		else
		{
			lostInfo = true;
			cout << "Failure !" << endl << endl << endl;
		}
	}
}


/////////////////*************** Remove boxes to reduce the bound **************////////////////////


//make a fusion of atMost1 boxes, when there can be only one zero in the union
void removeAtMost1Box(IMap &gradPot, IMap &deriv, vector<IVector> & res)
{
	int i = res.size()-1;
	while (i >= 0)
	{
		int j = res.size()-1;
		while (j > i )
		{
			IVector tmp(2);
			
			IVector unionB = unionBox(res[i], res[j]);
			if (diamVect(unionB) < 2*(diamVect(res[i])+diamVect(res[j])) )
			{	
				if (atMostOneZero(gradPot, deriv, unionB) )
				{
					res.erase(res.begin()+j);
					res[i] = unionB;
				}
			}
			j--;
		}
		i--;
	}
}


//make a fusion of atMost2 boxes, when there can be at most 2 zeros in the union
//X says if we want to remove atMost2 or atMost3
//if we cannot do the fusion of the boxes, but we can prove that in the union of 2 atMostX boxes
//there are at most Y = X+1 zeros, we replace the 2 atMostX boxes by 1 atMostY box
void removeAtMostX(IMap &gradPot, IMap &deriv, IMap &derivExp, vector<IVector> & res, vector<IVector> & resY,  int X)
{
	int i = res.size()-1;
	while (i >= 0)
	{
		int j = res.size()-1;
		while (j > i )
		{
			
			IVector unionB = unionBox(res[i], res[j]);

			if (diamVect(unionB) < minSizeBox + diamVect(res[i])+diamVect(res[j]) )
			{
				try
				{
					bool atMostX = false;
					if (X == 1)
						atMostX = atMostOneZero(gradPot, deriv, unionB);
					else if (X == 2)
						atMostX = atMostTwoZero(gradPot, deriv, unionB);
					else if (X == 3)
						atMostX = atMostThreeZero(gradPot, deriv, derivExp, unionB); //attention, give the good deriv
					if (atMostX )
					{
						res.erase(res.begin()+j);
						res[i] = unionB;
					}

					else if (X < 3)
					{
						bool atMostY = false; // Y = X+1, we try to replace 2 boxes with X zeros by one with Y
						if (X == 1)
							atMostY = atMostTwoZero(gradPot, deriv, unionB);
						else if (X == 2)
							atMostY = atMostThreeZero(gradPot, deriv, derivExp, unionB);

						if (atMostY)
						{
							res.erase(res.begin()+j); //erase j first because j > i
							res.erase(res.begin()+i);
							resY.push_back(unionB);
							j = i;
						}
					}
				}

				catch (exception& e)
				{
					#pragma omp critical (cerr)
						cerr << "I avoid quitting because of " << e.what() << endl;
				}
			}

			j--;
		}
		i--;
	}
}

//X,Y = 1,2,or 3, X < Y
//remove boxes B that contain at most X zeros if there exist a box C containing at most Y zeros
//and we can prove that the box containing B and C contains at most Y zeros
//replace box C by the box containing B and C 
void removeInterAtMostXY(IMap &gradPot, IMap &deriv, IMap &derivExp, vector<IVector> & atMostX, vector<IVector> & atMostY, int X, int Y)
{
	for (unsigned int i = 0; i < atMostY.size(); i++)
	{
		int j = atMostX.size();
		while (j>0)
		{
			j--;
			IVector tmp(2);

			IVector unionB = unionBox(atMostY[i], atMostX[j]);
			if (diamVect(unionB) < 2*(diamVect(atMostY[i])+diamVect(atMostX[j])) )
			{	
				try{
					bool unionBok = false;
					if (Y == 2)
						unionBok = atMostTwoZero(gradPot, deriv, unionB);
					else if (Y == 3)
						unionBok = atMostThreeZero(gradPot, deriv, derivExp, unionB);
					else
						cerr << "Attention, wrong parameter in removeInter" << endl;
					if (unionBok )
					{
						atMostX.erase(atMostX.begin()+j);
						atMostY[i] = unionB;
					}

				}
				catch (exception& e)
				{
					#pragma omp critical (cerr)
						cerr << "I avoid quitting because of " << e.what() << endl;
				}


			}
		}
	}
}

// If we can prove that the union of an atMost1 box and an atMost2 box has at most 3 zeros,
//we put the unionBox into the vector containing atMost3 boxes (without erasing the others).
void replace12by3(IMap &gradPot, IMap &deriv, IMap &derivExp, vector<IVector> & atMost1, vector<IVector> & atMost2, vector<IVector> & atMost3)
{
	int i = atMost1.size()-1;
	while (i >= 0)
	{
		int j = atMost2.size()-1;
		while (j >= 0 )
		{

			{	
				IVector unionB = unionBox(atMost1[i], atMost2[j]);
				try
				{
					if (atMostThreeZero(gradPot, deriv, derivExp, unionB) )
					{
						atMost3.push_back(unionB);
					}

				}
				catch (exception& e)
				{
					#pragma omp critical (cerr)
						cerr << "I avoid quitting because of " << e.what() << endl;
				}


			}
			j--;
		}
		i--;
	}
}

//replace the 1st box of atMost3 (if non empty) by a bigger box if it also contains at most 3 zeros.
//AtMostThreeZero can have some problems returning true if the box is too small.
void bigBoxAtMost3(IMap &gradPot, IMap &deriv, IMap &derivExp, vector<IVector> & atMost3)
{
	if (atMost3.size() > 0)
	{
		IVector bigBox = atMost3[0];
		interval increase(-0.015, 0.015);
		bigBox[0] += increase;
		bigBox[1] += increase;

		if (atMostThreeZero(gradPot, deriv, derivExp, bigBox) )
			atMost3[0] = bigBox;
	}
}



/////////////*********** Functions for the Newton Iteration ***********///////////////

//Compute the box N needed in the Newton Iteration 
//N = y+V where V contains all the solutions of F'(box)*v = -F(y)
//here, y is the middle of the box
IVector computeN(const IVector &box, IMap &gradPot, IMap &deriv)
{
	IVector y(2);
	IVector V(2);

	try
	{
		y[0] = box[0].mid();
		y[1] = box[1].mid();

		IMatrix A(2,2);
		IVector tmp = deriv(box);
		A[0][0] = tmp[0];
		A[1][0] = tmp [1];
		A[0][1] = tmp[2];
		A[1][1] = tmp [3];


		V = -capd::matrixAlgorithms::krawczykInverse(A)*gradPot(y);
	}
	catch (exception& e)
	{
		//cout << "Exception caught!\n" << e.what() << endl;
		V[0].setLeftBound(-20.); //represent infinity because we will always be in [-5,5]*[-5,5]
		V[0].setRightBound(20.);
		V[1].setLeftBound(-20.);
		V[1].setRightBound(20.);
	}

	return y+V;
}


//Alternative to N in the newton iteration
//Compute K for the Krawczyk method
IVector computeK(const IVector &box, IMap &gradPot, IMap &deriv)
{
	IVector m(2);
	m[0] = mid(box[0]);
	m[1] = mid(box[1]);

	IMatrix A(2,2);
	IVector tmp = deriv(m);
	A[0][0] = tmp[0];
	A[1][0] = tmp [1];
	A[0][1] = tmp[2];
	A[1][1] = tmp [3];

	IMatrix Y = capd::matrixAlgorithms::krawczykInverse(A);

	IMatrix Df(2,2);
	tmp = deriv(box);
	Df[0][0] = tmp[0];
	Df[1][0] = tmp [1];
	Df[0][1] = tmp[2];
	Df[1][1] = tmp [3];

	IMatrix I(2,2);
	I[0][0] = 1.;
	I[1][1] = 1.;

	IVector K = m-Y*gradPot(m)+ (I-Y*Df)*(box-m);
	return K;
}


//input : withN is true if we want tu use N, false if we want to use K
//output :  1 if there is a unique solution in box
//          0 if there are no solution in box
//         -1 if we have done too many iteration without result
//         -2 if the result is undefined (usually because f' is not invertible over the whole box)
//If there is a unique solution, a small box containing the solution is put in res
int NewtonIteration(const IVector &box, IMap &gradPot, IMap &deriv, IVector & res, bool withN)
{
	IVector N(2);
	if (withN)
		N = computeN(box, gradPot, deriv);
	else
		N = computeK(box, gradPot, deriv);

	IVector smallBox(2);
	bool inter = intersection(N, box, smallBox);
	IVector image = gradPot(box);
	res = smallBox;
	if (! inter || ! image[0].contains(0.) || ! image[1].contains(0.) )
		return 0;
	if (N[0].contains(box[0]) && N[1].contains(box[1]) )
		return -2;
	
	//we can have a solution
	int newton = -1;
	if (N[0].subsetInterior(box[0]) && N[1].subsetInterior(box[1]))
	{
		newton = 1;
	}

	if ( diam(smallBox[0]) + diam(smallBox[1]) < minDiamSize )
	{
		return newton;
	}
	else
	{
		int newton2 = NewtonIteration(smallBox, gradPot, deriv, res, withN);
		if (newton == 1)
			return 1;
		else
			return newton2;
	}
}


//Apply NewtonIteration recursively to try get 0 or 1
//Work only for boxes where we know there is at most 1 solution, 
//because we return 1 as soon as we find a solution
int recursiveNewton(const IVector &box, IMap &gradPot, IMap &deriv, IVector & res, bool withN)
{
	int newton = NewtonIteration(box, gradPot, deriv, res, withN);
	if (newton != -2)
		return newton;

	else
	{
		stack<IVector> waitingBoxes;
		splitBoxInStack(box, waitingBoxes);	

		while (!waitingBoxes.empty())
		{
			IVector topBox = waitingBoxes.top();
			waitingBoxes.pop();
		
			newton = NewtonIteration(topBox, gradPot, deriv, res, withN);

			if (newton == 1)
				return 1;

			else if (newton == 2 && diamVect(topBox) > minNewton)
				splitBoxInStack(topBox, waitingBoxes);				
		}

		return -2;
	}
}

//Check all atMost1 boxes with the Newton method.
//If we can prove the box has exactly one zero, remove the box in atMost1 and add it in posOfSol.
void transformAtMost1IntoGoodBoxes(IMap &gradPot, IMap &deriv, vector<IVector> &posOfSol, vector<IVector> &atMost1)
{
	int i = atMost1.size()-1;

	while (i >= 0)
	{
		IVector res(2);
		int newton = NewtonIteration(atMost1[i], gradPot, deriv, res, false); //recursiveNewton too long
		if (newton == 1)
		{
			posOfSol.push_back(res);
			atMost1.erase(atMost1.begin()+i);
		}

		else if (newton == 0)
			atMost1.erase(atMost1.begin()+i);
			
		i--;
	}

}


///////////*********** Global functions *************/////////////

//compute the total number of zeros, the boxes where we can compute bounds, and then remove boxes that are not
//usefull
int totalNoZero(const IVector &box, IMap &gradPot, IMap &deriv, IMap &derivExp, bool & lostInfo, vector<IVector> & res, vector<IVector> & atMost1, vector<IVector> & atMost2, vector<IVector> & atMost3)
{
	lostInfo = false;
	noZero(box, gradPot, deriv, derivExp, lostInfo, atMost1, atMost2, atMost3);
	int upperBound = atMost1.size()+2*atMost2.size()+3*atMost3.size();

	//take a lot of time to remove boxes in limit cases, and not useful if lostInfo == true
	if (lostInfo == false)
	{	

		removeAtMost1Box(gradPot, deriv, atMost1);
		removeAtMostX(gradPot, deriv, derivExp, atMost2, atMost3, 2);
		bigBoxAtMost3(gradPot, deriv, derivExp, atMost3);
		removeAtMostX(gradPot, deriv, derivExp, atMost3, atMost3, 3);
		removeInterAtMostXY(gradPot, deriv, derivExp, atMost1, atMost2, 1, 2);
		removeInterAtMostXY(gradPot, deriv, derivExp, atMost1, atMost3, 1, 3);
		removeInterAtMostXY(gradPot, deriv, derivExp, atMost2, atMost3, 2, 3);

		upperBound = res.size()+atMost1.size()+atMost2.size()*2+atMost3.size()*3;

		if (upperBound > 13)
		{
			removeAtMostX(gradPot, deriv, derivExp, atMost1, atMost2, 1);
			removeAtMostX(gradPot, deriv, derivExp, atMost2, atMost3, 2);
			bigBoxAtMost3(gradPot, deriv, derivExp, atMost3);
			removeAtMostX(gradPot, deriv, derivExp, atMost3, atMost3, 3);
			removeInterAtMostXY(gradPot, deriv, derivExp, atMost1, atMost2, 1, 2);
			replace12by3(gradPot, deriv, derivExp, atMost1, atMost2, atMost3);
			removeInterAtMostXY(gradPot, deriv, derivExp, atMost1, atMost3, 1, 3);
			removeInterAtMostXY(gradPot, deriv, derivExp, atMost2, atMost3, 2, 3);
			removeAtMostX(gradPot, deriv, derivExp, atMost3, atMost3, 3);

			upperBound = res.size()+atMost1.size()+atMost2.size()*2+atMost3.size()*3;
		}

		transformAtMost1IntoGoodBoxes(gradPot, deriv, res, atMost1);
	}
	return upperBound;
}

//compute the IMap gradPot and deriv from the three masses m1, m2 and m3
//and then compute an upper bound on the total number of zeros and all other infos like in totalNoZero
int createFunctionAndZeros(const interval &m1, const interval & m2, const interval & m3, const IVector &box, vector<IVector> & posOfZero, bool & lostInfo, vector<IVector> & atMost1, vector<IVector> & atMost2, vector<IVector> & atMost3, omp_lock_t &lock)
{
	interval sqrt3(3.);
	sqrt3 = sqrt(sqrt3);
	interval oneHalf(1.);
	oneHalf = oneHalf/2;
	int dimIn=2, dimOut=2, noParam=7;
	int dimIn2=2, dimOut2=4, noParam2=7;
	int upperBound = 0;

	interval Cx = (m1/3. - m2/6. - m3/6.)*sqrt3;
	interval Cy = (m2-m3)/2.;

	omp_set_lock(&lock);

	IMap gradPot(gradPotential,dimIn,dimOut,noParam, 3);
	IMap deriv(derivativeGradPotential,dimIn2,dimOut2,noParam2, 2);
	IMap derivExp(derivativeGradPotentialExp,dimIn2,dimOut2,noParam2, 2);

	gradPot.setParameter(0,Cx);
	gradPot.setParameter(1,Cy);
	gradPot.setParameter(2,m1);
	gradPot.setParameter(3,m2);
	gradPot.setParameter(4,m3);
	gradPot.setParameter(5,sqrt3);
	gradPot.setParameter(6,oneHalf);
	deriv.setParameter(0,Cx);
	deriv.setParameter(1,Cy);
	deriv.setParameter(2,m1);
	deriv.setParameter(3,m2);
	deriv.setParameter(4,m3);
	deriv.setParameter(5,sqrt3);
	deriv.setParameter(6,oneHalf);
	derivExp.setParameter(0,Cx);
	derivExp.setParameter(1,Cy);
	derivExp.setParameter(2,m1);
	derivExp.setParameter(3,m2);
	derivExp.setParameter(4,m3);
	derivExp.setParameter(5,sqrt3);
	derivExp.setParameter(6,oneHalf);

	lostInfo = false;

	omp_unset_lock(&lock);

	upperBound = totalNoZero(box, gradPot, deriv, derivExp, lostInfo, posOfZero, atMost1, atMost2, atMost3);

	return upperBound;
}


////////////////***************** Tests ***************////////////////////////

//compute a bound of the number of zeros for masses in boxes, not too close from m1 = 0
//write the points in different files according to the bound found
void testDifferentMasses(ofstream & notGood, ofstream & inf11sol, ofstream & e11sol, ofstream & e12sol, ofstream & e13sol, ofstream &more13)
{

	omp_lock_t lock;
	omp_init_lock(&lock);

	interval boxX(-5., 4.1);
	interval boxY(-5., 4.1);
	interval xcoeff[] = {boxX,boxY};
	IVector box(2,xcoeff);
	interval boxX2(-5., 4.);
	interval boxY2(-5., 4.);
	interval xcoeff2[] = {boxX2,boxY2};
	IVector box2(2,xcoeff2);


	#pragma omp parallel for schedule (dynamic)
	for (int i = 50; i < 334 ; i+=1)
	{
		double m1i = i;
		double m1d = m1i/1000.;
		interval m1(m1i, m1i+1.);
		m1 = m1/1000.;

		#pragma omp parallel for schedule (dynamic)
		for (int j = i; j <= (1000-i)/2+1; j +=1)
		{
			double m2i = j;
			double m2d = m2i/1000.;
			interval m2(m2i, m2i+1.);
			m2 = m2/1000.;
			interval m3 = 1.-m1-m2;


			vector<IVector> posOfZero;
			vector<IVector> atMost1;
			bool lostInfo = false;
			vector<IVector> atMost2;
			vector<IVector> atMost3;
			int upperBound = 0;

			upperBound =  createFunctionAndZeros(m1, m2, m3, box, posOfZero, lostInfo, atMost1, atMost2, atMost3, lock);
			/*if (upperBound > 13)
			{
				lostInfo = false;
				posOfZero.clear();
				atMost1.clear();
				atMost2.clear();
				atMost3.clear();
				upperBound = createFunctionAndZeros(m1, m2, m3, box2, posOfZero, lostInfo, atMost1, atMost2, atMost3, lock);
			}*/

			#pragma omp critical (cout)
			{
				cout << "m1 = " << m1 << ", m2 = " << m2 << " and m3 = " << m3 << endl;
				cout << "lost info : " << lostInfo << endl;
				cout << "Number of zeros : " << posOfZero.size() << endl;
				cout << "number of atMost1 boxes = " << atMost1.size() << endl;
				cout << "number of atMost2 boxes = " << atMost2.size() << endl;
				cout << "number of atMost3 boxes = " << atMost3.size() << endl;
				cout << "Number max of zeros : " << upperBound << endl << endl << endl;
			}

			double x = sqrt(3)/3.*(m2d-m1d); //coordinates of the corresponding point in the triangle of masses
			double y = 1.-m1d-m2d;

			if (lostInfo == true)
			{
				#pragma omp critical (notGood)
					notGood << x << " " << y << endl;
			}

			else if (upperBound < 11)
			{
				#pragma omp critical (inf11)
					inf11sol << x << " " << y << " " << upperBound << endl;
			}

			else if (upperBound == 11)
			{
				#pragma omp critical (e11sol)
					e11sol << x << " " << y << endl;

			}

			else if (upperBound == 12)
			{
				#pragma omp critical (e12sol)
					e12sol << x << " " << y << endl;
			}

			else if (upperBound == 13)
			{
				#pragma omp critical (e13sol)
					e13sol << x << " " << y << endl;
			}

			else
			{
				#pragma omp critical (more13)
					more13 << x << " " << y << " " << upperBound << endl;
			}

		}
	}

	omp_destroy_lock(&lock);

}



///////////////****************** Main ******************////////////////////

int main(int argc, char** argv)
{
	
	ofstream notGood("data/notGood.dat", ios::out | ios::trunc);
	ofstream inf11sol("data/inf11sol.dat", ios::out | ios::trunc);
	ofstream more13("data/more13sol.dat", ios::out | ios::trunc);
	ofstream e11sol("data/11sol.dat", ios::out | ios::trunc);
	ofstream e12sol("data/12sol.dat", ios::out | ios::trunc);
	ofstream e13sol("data/13sol.dat", ios::out | ios::trunc);

	testDifferentMasses(notGood, inf11sol, e11sol, e12sol, e13sol, more13);

	notGood.close();
	inf11sol.close();
	e11sol.close();
	e12sol.close();
	e13sol.close();
	more13.close();

	return 0;
}

	







