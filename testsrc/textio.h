/*
 *
 * textio.h
 * Assortment of rudimentary text/ascii file input and output functions.
 *
 * K Erik J Olofsson, November 2014 / December 2015
 */

#ifndef __TEXTIO_H__
#define __TEXTIO_H__

#define TEXTIO_MAXBUFFER 5000
#define TEXTIO_COLUMNMAJOR +1
#define TEXTIO_ROWMAJOR -1

/* write a file to disk with one integer number per line (like a column) */
int textio_write_integer_array_column(char *filename,int *a,int n) {
	int jj;
	FILE *pfile = NULL;
	pfile = fopen(filename,"w");
	if (!pfile) {
		return 0;
	}
	for (jj=0;jj<n;jj++) {
		fprintf(pfile,"%i\n",a[jj]);
	}
	fclose(pfile);
	return 1;
}

/* write a file to disk with a single colums of numbers of doubles */
int textio_write_double_array_column(char *filename,double *a,int n) {
	int jj;
	FILE *pfile = NULL;
	pfile = fopen(filename,"w");
	if (!pfile) {
		return 0;
	}
	for (jj=0;jj<n;jj++) {
		fprintf(pfile,"%f\n",a[jj]);
	}
	fclose(pfile);
	return 1;
}

/* write a matrix to a file (TODO formatspec as parameter) */
int textio_write_double_array_matrix(char *filename,double *a,int m,int n,char *formatspec) {
	static char default_formatspec[]="%.10e";
	static char str1[8];
	static char str2[8];
	if (formatspec!=NULL) {
		sprintf(str1,"%s ",formatspec);
		sprintf(str2,"%s\n",formatspec);
	} else {
		sprintf(str1,"%s ",default_formatspec);
		sprintf(str2,"%s\n",default_formatspec);
	}
	int ii,jj;
	FILE *pfile = NULL;
	pfile = fopen(filename,"w");
	if (!pfile) {
		return 0;
	}
	for (ii=0;ii<m;ii++) {
		for (jj=0;jj<(n-1);jj++) {
			fprintf(pfile,str1,a[jj*m+ii]);
		}
		fprintf(pfile,str2,a[jj*m+ii]);
	}
	fclose(pfile);
	return 1;
}

/* write a file to disk with integers row by row, c columns */
int textio_write_integer_rows(char *filename,int *a,int c,int n) {
	int jj,cc;
	FILE *pfile = NULL;
	pfile = fopen(filename,"w");
	if (!pfile) {
		return 0;
	}
	cc = 0;
	for (jj=0;jj<n;jj++) {
		fprintf(pfile,"%i ",a[jj]+1);
		cc++;
		if (cc == c) {
			fprintf(pfile,"\n");
			cc = 0;
		}
	}
	fclose(pfile);
	return 1;
}

int textio_preread_textfile(char *filename,int *numvalues) {
	/* Open file and scan through to count how many numbers it contains, then close the file. */
	assert(filename!=NULL);
	assert(numvalues!=NULL);
	
	*numvalues = 0;
	FILE *fp = NULL;
	fp = fopen(filename,"r");
	if (!fp) {
		return 0;
	}
	
	int ret;
	int counter = 0;
	double dmy;
	while (1) {
		ret = fscanf(fp,"%lf",&dmy);
		if (ret!=1) {
			break;
		}
		counter++;
	}

	fclose(fp);
	*numvalues = counter;
	return 1;
}

int textio_read_textfile_array(char *filename,int *numvalues,double *a,int maxvalues) {
	assert(filename!=NULL);
	assert(numvalues!=NULL);
	assert(a!=NULL);
	assert(maxvalues>0);
	
	*numvalues = 0;
	FILE *fp = NULL;
	fp = fopen(filename,"r");
	if (!fp) {
		return 0;
	}
	
	int ret;
	int counter = 0;
	while (counter<maxvalues) {
		ret = fscanf(fp,"%lf",&a[counter]);
		if (ret!=1) {
			break;
		}
		counter++;
	}
	
	fclose(fp);
	*numvalues = counter;
	return 1;
}

/* Read a matrix from a textfile; only count elements if buf==NULL, otherwise store elements in buf */
int textio_read_table_file(char *filename,int *m,int *n,double *buf,int maxbuf,int colStride,int rowStride) {
	static char linebuffer[TEXTIO_MAXBUFFER];
	*m=0; *n=0;
	FILE *fp = NULL;
	fp = fopen(filename,"r");
	if (!fp) {
		return 0;
	}
	
	int ret,idx;
	int totalcounter = 0;
	int rowcounter = 0;
	int columncounter = 0;
	int firstcolumncount = 0;
	double dmy;
	char *tmp,*pch;
	
	while (1) {
		tmp=fgets(linebuffer,TEXTIO_MAXBUFFER,fp);
		if (tmp==NULL) {
			break;
		}
		rowcounter++;
		columncounter=0;
		pch=strtok(tmp," ");
		while (pch!=NULL) {
			ret=sscanf(pch,"%lf",&dmy);
			if (ret!=1) break;
			/* printf("%s : %.6f\n",pch,dmy); */
			columncounter++;
			if (buf!=NULL) { /* calculate the index for storage based on colStride,rowStride ints */
				idx=colStride*(columncounter-1)+rowStride*(rowcounter-1);
				if (idx<maxbuf && idx>=0) buf[idx]=dmy;
			}
			totalcounter++;
			pch=strtok(NULL," ");
		}
		if (rowcounter==1) {
			firstcolumncount=columncounter;
		}
		if (columncounter!=firstcolumncount) {
			fclose(fp);
			return -rowcounter;
			/*break;*/	/* need to indicate error here */
		}
		/*printf("[#%i]: %i\n",rowcounter,columncounter); */
	}
	*m=rowcounter;
	*n=firstcolumncount;
	/*printf("total=%i\n",totalcounter);*/
	fclose(fp);
	return totalcounter;
}

/* This code should pre-read the file: check that M*N=totalcount and all rows have the same length; then
   assign the column-stride and row-stride and fill the array to a newly allocated memory slot. */
int textio_read_matrix_utility(char *filename,double **dat,int *m,int *n,int colmaj) {
	int ret,rows,cols;
	int ret2,rows2,cols2;
	*dat=NULL;
	ret = textio_read_table_file(filename,&rows,&cols,NULL,-1,-1,-1);
	if (ret<=0) {
		return ret;	/* failure code */
	}
	if (ret!=rows*cols) {
		return 0;	/* also a failure code */
	}
	*dat=(double*)malloc(sizeof(double)*ret);	/* Allocate memory of size ret*sizeof(double) */
	if (*dat==NULL) {
		return 0;
	}
	if (colmaj>0) {
		ret2 = textio_read_table_file(filename,&rows2,&cols2,*dat,ret,rows,1); /* set colStride=rows, rowStride=1 */
	} else {
		/* not yet tested */
		ret2 = textio_read_table_file(filename,&rows2,&cols2,*dat,ret,1,cols); /* set colStride=1, rowStride=cols */
	}
	if (ret2!=ret || ret2!=rows2*cols2) {
		return 0;
	}
	*m=rows2;
	*n=cols2;
	return ret2;
}

#endif
