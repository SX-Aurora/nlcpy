#!/usr/bin/perl
#
# * The source code in this file is developed independently by NEC Corporation.
#
# # NLCPy License #
# 
#     Copyright (c) 2020-2021 NEC Corporation
#     All rights reserved.
#     
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither NEC Corporation nor the names of its contributors may be
#       used to endorse or promote products derived from this software
#       without specific prior written permission.
#     
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#     ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#     WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#     ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#     (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#     SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

use strict;
use File::Basename;
use warnings qw(FATAL all);
our ($script);
our (%macro, @macro, @dtos, $asm, %branch, @branch);
our ($infile, $outfile);
our ($op, $vors, @ar, $type1, $type2, $ufunc_op, $op_save);
our $mac = qr/[A-Za-z_]\w*/;    # regular expression for macro names
our (%dtag_table,%operator_table, %irregular_intrinsic_table, %c_intrinsic_types_table);
$script =$0;

######################### Parse command-line arguments ########################

while ($_ = shift) {
    /^-D($mac)=(["']?)(.*)\2/                    ?  $macro{$1} = $3     :
    /^-D$/ && shift =~ /^($mac)=(["']?)(.*)\2/   ?  $macro{$1} = $3     :
    /^-D($mac)/ || /^-D$/ && shift =~ /^($mac)/  ?  $macro{$1} = 1      :
    /^-U($mac)/ || /^-U$/ && shift =~ /^($mac)/  ?  delete($macro{$1})  :
    /^-./                                        ?  usage()             :
    !defined($infile)                            ?  $infile = $_        :
    !defined($outfile)                           ?  $outfile = $_       :
    usage();
}

usage() unless defined $infile && defined $outfile;

########################## Option check #######################################

($macro{FILENAME}) = ($outfile =~ m%^(?:.*/)?([^\.]*)%) or
   name_error("Cannot determine function name from name \"$outfile.\"");

if      ($macro{FILENAME} =~ m%^nlcpy_(.*)__(.*)__(.*)__(.*)$% ) {
  # ternary_operator
  $op    = $1;
  $ar[0] = $2;
  $ar[1] = $3;
  $ar[2] = $4;
} elsif ($macro{FILENAME} =~ m%^nlcpy_(.*)__(.*)__(.*)$% ) {
  # binary_operator
  $op    = $1;
  $ar[0] = $2;
  $ar[1] = $3;
} elsif ($macro{FILENAME} =~ m%^nlcpy_(.*)__(.*)$% ) {
  # unary_operator
  $op    = $1;
  $ar[0] = $2;
} elsif ($macro{FILENAME} =~ m%^nlcpy_(.*)$% ) {
  $op    = $1;
} else {
   name_error("Cannot determine a operation from name \"$outfile.\"");
}
#
# Add operator macro name
#
$op_save = $op;
$ufunc_op   = 0;
if ( $op =~ /_reduce$/) {
   $op =~ s/_reduce$//;
   $ufunc_op = 1;
} elsif ( $op =~ /_accumulate$/) {
   $op =~ s/_accumulate$//;
   $ufunc_op = 1;
} elsif ( $op =~ /_reduceat$/) {
   $op =~ s/_reduceat$//;
   $ufunc_op = 1;
} elsif ( $op =~ /_outer$/) {
   $op =~ s/_outer$//;
   $ufunc_op = 1;
} elsif ( $op =~ /_at$/) {
   $op =~ s/_at$//;
   $ufunc_op = 1;
}
$macro{$op_save} = $op_save;
#
################################## Open files #################################

$infile = openfile($infile, "");

!defined($_ = (stat($outfile))[1]) or $_ != (stat($infile))[1]
    or die "Input file and output file cannot be the same\n";

open(OUTFILE,">$outfile") or die "Cannot open $outfile: $!\n";
print OUTFILE "/* DO NOT EDIT THIS: This file was automatically generated. */\n\n";

$SIG{__DIE__} = sub { unlink $outfile };   # rm outfile if preprocessor dies

create_table(%dtag_table, %operator_table, %irregular_intrinsic_table, %c_intrinsic_types_table);

### set output datatypes defined by %dtag_table ###
if (ref($dtag_table{$op})) {
   if (!defined($dtag_table{$op}->{'dtype'})) {
      my @out_ref = @{$dtag_table{$op}->{'out'}};
      for (my $i=0; $i<=$#out_ref; $i++){
         my @dt = split(/,/,$out_ref[$i]);
         foreach my $dtag (@dt){
            $dtag =~ s/\s//g;
            $macro{"DTAG_"."$dtag"}  = 1;
            $macro{"DTAG_OUT_"."$dtag"} = 1;
         }
      }
   } else {
      my @dtag_ref = @{$dtag_table{$op}->{'dtype'}};
      for (my $i=0; $i<=$#dtag_ref; $i++){
         my @dt = split(/,/,$dtag_ref[$i]);
         foreach my $dtag (@dt){
            $dtag =~ s/\s//g;
            $macro{"DTAG_"."$dtag"} = 1;
         }
      }
      my @out_ref = @{$dtag_table{$op}->{'out'}};
      for (my $i=0; $i<=$#out_ref; $i++){
         my @dt = split(/,/,$out_ref[$i]);
         foreach my $dtag (@dt){
            $dtag =~ s/\s//g;
            $macro{"DTAG_OUT_"."$dtag"} = 1;
         }
      }
   }
}

# Do preprocess for *.c.master2
my $infile2 = "$$infile" . "2";
if (-f $infile2 ) {
   @ar=();
   if (ref($dtag_table{$op})) {
      my @in_ref = @{$dtag_table{$op}->{'in'}};
      my $ary = $#in_ref+1;
      if ( $op_save =~ /_reduce$/ or  $op_save =~ /_reduceat$/ or  $op_save =~ /_accumulate$/) { $ary-- }
      if ($ary == 1) {
         my @dt1 = split(/,/,$in_ref[0]);
         foreach my $dtag1 (@dt1){
            $dtag1 =~ s/\s//g;
            $ar[0] = $dtag1;
            $macro{"DTAG1_"."$dtag1"} = 1;
            preprocess(openfile($infile2));
            delete $macro{"DTAG1_"."$dtag1"};
         }
      } elsif ($ary == 2) {
         my @dt1 = split(/,/,$in_ref[0]);
         foreach my $dtag1 (@dt1){
            $dtag1 =~ s/\s//g;
            $ar[0] = $dtag1;
            $macro{"DTAG1_"."$dtag1"} = 1;
            my @dt2 = split(/,/,$in_ref[1]);
            foreach my $dtag2 (@dt2){
               $dtag2 =~ s/\s//g;
               $ar[1] = $dtag2;
               $macro{"DTAG2_"."$dtag2"} = 1;
               preprocess(openfile($infile2));
               delete $macro{"DTAG2_"."$dtag2"};
            }
            delete $macro{"DTAG1_"."$dtag1"};
         }
      } elsif ($ary == 3) {
         my @dt1 = split(/,/,$in_ref[0]);
         foreach my $dtag1 (@dt1){
            $dtag1 =~ s/\s//g;
            $ar[0] = $dtag1;
            $macro{"DTAG1_"."$dtag1"} = 1;
            my @dt2 = split(/,/,$in_ref[1]);
            foreach my $dtag2 (@dt2){
               $dtag2 =~ s/\s//g;
               $ar[1] = $dtag2;
               $macro{"DTAG2_"."$dtag2"} = 1;
               my @dt3 = split(/,/,$in_ref[2]);
               foreach my $dtag3 (@dt3){
                  $dtag3 =~ s/\s//g;
                  $ar[2] = $dtag3;
                  $macro{"DTAG3_"."$dtag3"} = 1;
                  preprocess(openfile($infile2));
                  delete $macro{"DTAG2_"."$dtag2"};
               }
               delete $macro{"DTAG2_"."$dtag2"};
            }
            delete $macro{"DTAG1_"."$dtag1"};
         }
      } else {
         die "The number of operands is greater than 3 (Not supported).\nPlease contact R. Ogata\n";
      }
   }
}

# Do preprocess for *.c.master
preprocess($infile);

if (-f $infile2 ) {
  print "$outfile is generated from $$infile and $infile2\.\n";
} else {
  print "$outfile is generated from $$infile\.\n";
}

###############################################################################

our ($file,$line);
sub preprocess {
    local ($file,$line) = ($_[0], 0);
    my @cond = (1);
    my @vars = ();
    my @ds   = ();
    my @cases = ();
    my @buff = ();
    my @elseflag = ();
    my $buffering=0;
    my $begin_counter=0;
    my $depth=-1;
    my $default_case = "default:{\n}}\n";
    my $cond_depth=1;

    for ($line=1; <$file>; $line++) {

        if (/^\s*#\s*begin_switch\s*\b/ or /^\s*#\s*end_switch\b/) {  # begin_switch or end_switch
            next if ! $cond[0];
            $#ds + 1 or die "Undefined define_switch on $$file: $line\n";
            if (/^\s*#\s*begin_switch\s*\b/) {
                $begin_counter++;
                $buffering=1;
            } elsif (/^\s*#\s*end_switch\b/) {  # begin_switch or end_switch
                # dump
                my $tmpfile = "tmp_$line.txt";
                open(my $fh,">$tmpfile") or die "Cannot open $tmpfile: $!\n";
                foreach my $tmpline (@buff){
                    print $fh $tmpline;
                }
                @buff = (); # reset
                close($fh);
                # genarete codes between begin_switch and end_switch   
                print OUTFILE "uint64_t passed = 0;\n";
                if ($#vars+1==1) {
                    for (my $i=0; $i<=$#cases; $i++){
                        my @dtags1 = split(/,/,$cases[0]);
                        print OUTFILE "switch ($vars[0]) {\n";
                        foreach my $dtag1 (@dtags1){
                            $ar[0] = $dtag1;
                            $macro{"DTAG1_"."$dtag1"} = 1;
                            print OUTFILE "case ve_"."$dtag1".":{\n";
                            preprocess(openfile($tmpfile, "$$file: $line: "));
                            print OUTFILE "passed++;\nbreak;}\n";
                            delete $macro{"DTAG1_"."$dtag1"};
                        }
                        print OUTFILE $default_case;
                    }
                } elsif ($#vars+1==2) {
                    for (my $i=0; $i<=$#cases; $i+=2){
                        my @dtags1 = split(/,/,$cases[$i]);
                        my @dtags2 = split(/,/,$cases[$i+1]);
                        print OUTFILE "switch ($vars[0]) {\n";
                        foreach my $dtag1 (@dtags1){
                            $ar[0] = $dtag1;
                            $macro{"DTAG1_"."$dtag1"} = 1;
                            print OUTFILE "case ve_"."$dtag1".":{\n";
                            print OUTFILE "switch ($vars[1]) {\n";
                            foreach my $dtag2 (@dtags2){
                                $ar[1] = $dtag2;
                                print OUTFILE "case ve_"."$dtag2".":{\n";
                                preprocess(openfile($tmpfile, "$$file: $line: "));
                                print OUTFILE "passed++;\nbreak;}\n";
                                delete $macro{"DTAG2_"."$dtag2"};
                            }
                            print OUTFILE $default_case;
                            print OUTFILE "break;}\n";
                            delete $macro{"DTAG1_"."$dtag1"};
                        }
                        print OUTFILE $default_case;
                    }
                } elsif ($#vars+1==3) {
                    for (my $i=0; $i<=$#cases; $i+=3){
                        my @dtags1 = split(/,/,$cases[$i]);
                        my @dtags2 = split(/,/,$cases[$i+1]);
                        my @dtags3 = split(/,/,$cases[$i+2]);
                        print OUTFILE "switch ($vars[0]) {\n";
                        foreach my $dtag1 (@dtags1){
                            $ar[0] = $dtag1;
                            $macro{"DTAG1_"."$dtag1"} = 1;
                            print OUTFILE "case ve_"."$dtag1".":{\n";
                            print OUTFILE "switch ($vars[1]) {\n";
                            foreach my $dtag2 (@dtags2){
                                $ar[1] = $dtag2;
                                $macro{"DTAG2_"."$dtag2"} = 1;
                                print OUTFILE "case ve_"."$dtag2".":{\n";
                                print OUTFILE "switch ($vars[2]) {\n";
                                foreach my $dtag3 (@dtags3){
                                    $ar[2] = $dtag3;
                                    $macro{"DTAG3_"."$dtag3"} = 1;
                                    print OUTFILE "case ve_"."$dtag3".":{\n";
                                    preprocess(openfile($tmpfile, "$$file: $line: "));
                                    print OUTFILE "passed++;\nbreak;}\n";
                                    delete $macro{"DTAG3_"."$dtag3"};
                                }
                                print OUTFILE $default_case;
                                print OUTFILE "break;}\n";
                                delete $macro{"DTAG2_"."$dtag2"};
                            }
                            print OUTFILE $default_case;
                            print OUTFILE "break;}\n";
                            delete $macro{"DTAG1_"."$dtag1"};
                        }
                        print OUTFILE $default_case;
                    }
                } else {
                    die "The number of variables of #define_switch is greater than 3 (Not supported).\nPlease contact R. Ogata\n"; close $file;
                }
                print OUTFILE "if (passed!=1ULL) return NLCPY_ERROR_DTYPE;\n";
                unlink $tmpfile;
                $buffering=0;
            }
        } elsif ($cond[0] and $buffering) {
########## buffuring codes between begin_switch and end_switch ########
              push(@buff, $_);
#######################################################

        } elsif (my ($def,$name) = /^\s*#\s*if(n?def)\s+($mac)\b/) {    # if[n]def
            unshift(@cond, $cond[0] && ($def =~ /n/ != exists $macro{$name}));
            $elseflag[++$depth] = $cond[0];
        } elsif (my ($expr) = /^\s*#\s*if\s+(.*)/) {
            unshift(@cond, $cond[0] && evalif($expr) ? 1 : 0);
            $elseflag[++$depth] = $cond[0];
        } elsif (my ($expr2) = /^\s*#\s*elif\s+(.*)/) {
            shift @cond;
            unshift(@cond, $cond[0] && !$elseflag[$depth] && evalif($expr2) ? 1 : 0);
            $elseflag[$depth] = $cond[0] || $elseflag[$depth];
        } elsif (/^\s*#\s*else\b/) {
            $#cond or die "Unmatched #else on $$file: $line\n";
            shift @cond;
            unshift(@cond, $cond[0] && !$elseflag[$depth] ? 1 : 0);
        } elsif (/^\s*#\s*endif\b/) {
            $#cond or die "Unmatched #endif on $$file: $line\n";
            $depth--;
            shift @cond;
        } elsif ($cond[0]) {                              # If code is included
            if (/^\s*#\s*define\s+($mac)\s+(.*)/) {                #define
                $macro{$1} = $2;
                @macro = ();
            } elsif (/^\s*#\s*undef\s+($mac)/) {                   #undef
                delete $macro{$1};
                @macro = ();
            } elsif (/^\s*#\s*include\s+"([^"\n]+)"/) {            #include
		my $incfile = $1;
                preprocess(openfile($incfile, "$$file: $line: "));
		# <--
            } elsif (my ($err) = /^\s*#\s*error\s+(("?).*\2)/) {   #error
                $err =~ s/\s+$//;
                $err .= " " if length $err;
                die "#error ${err}on $$file: $line\n";
            } elsif (/^\s*#\s*define_switch\s*\((.*)\)\s*$/) {    # define_switch
                $begin_counter=0;  # clear
                @ds = ();
                @ds = split(/@/,$1);
                my $ii = 0;
                foreach my $d (@ds){
                    $d =~ s/\s//g;
                    $d =~ s/,,/,/g; # delete empty
                    if ($d =~ /(.+):(.+)/) {
                         # ex.) x->dtype:i32,i64  --> $1=x->dtype, $2=i32,i64 
                         push(@vars,$1);
                         push(@cases,$2);
                    }elsif ($d =~ /(.+)/) {
                         # set input datatypes defined by %dtag_table
                         push(@vars,$1);
                         if (ref($dtag_table{$op})) {
                             my @in_ref  = @{$dtag_table{$op}->{'in'}};
                             if ($ii>$#in_ref){
                                die "Mismatch the dimensions \#define_switch and \$dtag_table{$op}.\n";
                             }
                             my $typ = $in_ref[$ii];
                             $typ =~ s/\s//g;
                             $typ =~ s/,,/,/g; # delete empty
                             push(@cases,"$typ".",");
                         } else {
                             print "$_";
                             die "Not found $op in \$dtag_table. ($$file: $line)\n";
                         }
                    } else {
                         die "Parse error on $$file: $line\n";
                    }
                    $ii++;
                }
            } elsif (/^\s*#\s*adddef_switch\s*\((.*)\)\s*$/) {    # adddef_switch
                my $d = $1;
                $d =~ s/\s//g;
                push(@cases, split(/@/,$d));

            
            } else {
        	# <--
                macro();
                doc();
                print OUTFILE;
            }
        }
    }

    die "Missing #endif at EOF on $$file\n" if $#cond;

    close $file;
}

##################################### Sed ####################################
sub doc {
    # escape character
    s/^@//;
    my ($lop, $rop);
    my $ary = $#ar+1;

    s/\@OPERATOR_NAME\@/$op/g;
    s/FILENAME/$macro{FILENAME}/g;

    if ($ary==1){
      # Unary Operator
      $rop = $ar[0];
      $type1 = tag_to_Dtype($rop);
      s/\@TYPE1\@/$type1/g;
      s/\@DTAG1\@/$rop/g;
    } elsif ($ary==2){
      # Binary Operator
      $lop = $ar[0];
      $rop = $ar[1];
      $type1 = tag_to_Dtype($lop);
      $type2 = tag_to_Dtype($rop);
      s/\@TYPE1\@/$type1/g;
      s/\@TYPE2\@/$type2/g;
      s/\@DTAG1\@/$lop/g;
      s/\@DTAG2\@/$rop/g;
    } elsif ($ary==3){
      die "The number of operands is greater than 3 (Not implemented).\nPlease contact R. Ogata\n"; close $file;
    }
###
    if (/(.*)\@CAST_OPERATOR\@\(\s*([^,]*),\s*([^,]*),\s*([^,]*)\)(.*)/) {
       my $idt      = $1;       # indent
       my $op1      = $2;       # source
       my $src_dtag = $3;       # data type tag of source
       my $tar_dtag = $4;       # data type tag of target
       my $ftr      = $5;       # footer
       $_ = cast_operator($op,$idt,$op1,$src_dtag,$tar_dtag);

    } elsif (/(.*)\@UNARY_OPERATOR\@\(\s*([^,]*),\s*([^,]*),\s*([^,]*)\)(.*)/) {
       my $idt      = $1;       # indent
       my $op1      = $2;       # source
       my $op2      = $3;       # target
       my $op2_dtag = $4;       # data type tag of target
       my $ftr      = $5;       # footer
       my $op1_dtag = $ar[0];
       my $ari_dtag = $op2_dtag;# data type tag of arithmetics
       $_ = unary_operator($op,$ufunc_op,$idt,$op1,$op1_dtag,$op2,$op2_dtag,$ari_dtag);

    } elsif (/(.*)\@UNARY_OPERATOR_WITH_DTYPES\@\(\s*([^,]*),\s*([^,]*),\s*([^,]*),\s*([^,]*)\)(.*)/) {
       my $idt      = $1;       # indent
       my $op1      = $2;       # source
       my $op2      = $3;       # target
       my $op2_dtag = $4;       # data type tag of target
       my $ari_dtag = $5;       # data type tag of arithmetics
       my $ftr      = $6;       # footer
       my $op1_dtag = $ar[0];
       $_ = unary_operator($op,$ufunc_op,$idt,$op1,$op1_dtag,$op2,$op2_dtag,$ari_dtag);

    } elsif (/(.*)\@BINARY_OPERATOR\@\(\s*([^,]*),\s*([^,]*),\s*([^,]*),\s*([^,]*)\)(.*)/) {
       my $idt      = $1;       # indent
       my $op1      = $2;       # left operand
       my $op2      = $3;       # right operand
       my $op3      = $4;       # target
       my $op3_dtag = $5;       # data type tag of target
       my $ftr      = $6;       # footer
       my $op1_dtag = $ar[0];
       my $op2_dtag = $ar[1];
       my $ari_dtag = $op3_dtag;# data type tag of arithmetics
       $_ = binary_operator($op,$ufunc_op,$idt,$op1,$op1_dtag,$op2,$op2_dtag,$op3,$op3_dtag,$ari_dtag);

    } elsif (/(.*)\@BINARY_OPERATOR_WITH_DTYPES\@\(\s*([^,]*),\s*([^,]*),\s*([^,]*),\s*([^,]*),\s*([^,]*),\s*([^,]*),\s*([^,]*)\)(.*)/) {
       my $idt      = $1;  # indent
       my $op1      = $2;  # left operand
       my $op1_dtag = $3;  # data type of left operand
       my $op2      = $4;  # right operand
       my $op2_dtag = $5;  # data type of right operand
       my $op3      = $6;  # target
       my $op3_dtag = $7;  # data type tag of output
       my $ari_dtag = $8;  # data type tag of arithmetics
       my $ftr      = $9;  # footer
       $_ = binary_operator($op,$ufunc_op,$idt,$op1,$op1_dtag,$op2,$op2_dtag,$op3,$op3_dtag,$ari_dtag);

    }
###
}

############################ Apply macros and rules ###########################

sub macro {
    # escape character
    if(/^@/) {return;}

    if (!@macro) {      # Recompile @macro each time %macro changes
        @macro = map [qr/\b($_)\b/, '$macro{$1}'], keys %macro;
        push @macro, @dtos if exists $macro{SINGLE};
    }
    my @mac = @macro;   # Apply regexps, applying each one at most once
    for (my $repeat=1; $repeat--;) {
        for my $r (grep defined, @mac) {
            undef $r, $repeat=1 if s/$$r[0]/qq["$$r[1]"]/gee;
        }
    }
}

###############################################################################
#### For bug protection, preprocessor directives are not allowed to have ######
#### spaces before or after #'s because then comments look like directives ####
###############################################################################

sub die_if_spaces {
    die "$$file: $line: Leading spaces not allowed in preprocessor directive\n"
        if /^(\s+#|#\s+\S)/;
}

############################### Evaluate #if ##################################

sub evalif {
    local ($_) = @_;
    local $SIG{__WARN__} = sub { };
    s#/\*.*?\*/##g;
    s/defined\s*\(\s*($mac)\s*\)/0+exists $macro{$1}/ge;
    s/defined\s+($mac)/0+exists $macro{$1}/ge;
    $_ = eval;
    s/($mac)/exists $macro{$1} ? $macro{$1} : 0/ge;
    $_ = eval;
    return $_ unless $@;
    $@ =~ s/at \(eval .*//;
    die "$$file: $line: $@\n";
}

################################## Open files #################################

sub openfile {
    local *FILE;
    ${*FILE{SCALAR}} = $_[0];     # Sets $FILE to filename in fake glob
    open(FILE, "<$_[0]") or die "$_[1]Cannot open $_[0]: $!\n";
    return *FILE;
}

##################################### Usage ###################################

sub usage {die "Usage: $0 <infile> <outfile>\n";}

################################ Unary Operator ###############################
sub cast_operator{
   my $op      = $_[0];
   my $idt     = $_[1];
   my $op1     = $_[2];
   my $src_dtag= $_[3];
   my $tar_dtag= $_[4];
   #
   my $src_dtype = tag_to_Dtype($src_dtag);
   my $tar_dtype = tag_to_Atype($tar_dtag);
   
   my $bdy;
   if (/(\@cast_Bint\d\@)(\(\s*[^,)]*\)+?)/) {
     my $cast = $1;
     my $arg  = $2;
     $bdy = replace_cast_Bint($_,$src_dtag,$cast,$arg);
   } else {
     # others case
     $bdy = "($tar_dtype)($op1)";
   }
   return "$idt"."$bdy".";\n";
}

sub unary_operator{
   my $op      = $_[0];
   my $ufunc_op= $_[1];
   my $idt     = $_[2];
   my $op1     = $_[3];
   my $op1_dtag= $_[4];
   my $op2     = $_[5];
   my $op2_dtag= $_[6];
   my $ari_dtag= $_[7];
   my ($fn,$bdy,$op1_dtype,$op2_dtype,$ari_dtype,$extype1,$extype2);
   #
   $op1_dtype = tag_to_Dtype($op1_dtag);
   $op2_dtype = tag_to_Dtype($op2_dtag);
   $ari_dtype = tag_to_Atype($ari_dtag);
   #
   $fn = get_unary_intrinsic_name($op,$type1,$ari_dtype);
   if (exists($operator_table{$op})) {
      my @arg = ($op,$ari_dtype,$op1_dtype,$op1_dtype);
      $bdy = ref_operator_table(\%operator_table,\@arg);

      if ( $ufunc_op==1 ) {
          # swap
          $bdy =~ s/\@op1\@/\@op3\@/g;
          $bdy =~ s/\@op2\@/\@op1\@/g;
          $bdy =~ s/\@op3\@/\@op2\@/g;
      }

   } elsif( defined($fn) ) {
      if ( $ufunc_op!=1 ) {
         $extype1 = ${$c_intrinsic_types_table{$fn}}[0];
         if (defined($extype1) and $op1_dtype ne $extype1) {
             # do casting
             $bdy = '@op2@ = '."$fn(($extype1)(\@op1\@));";
         } else {
             $bdy = '@op2@ = '."$fn(\@op1\@);";
         }
      } else {
         # ufunc.reduce, ufunc.accumulate, etc.
         $extype1 = ${$c_intrinsic_types_table{$fn}}[0];
         $extype2 = ${$c_intrinsic_types_table{$fn}}[1];
         $bdy = "\@op2\@ = $fn(";
         if (defined($extype2) and $op2_dtype ne $extype2) {
             # do casting
             $bdy .= "($extype2)(\@op2\@),";
         } else {
             $bdy .= "\@op2\@,";
         }
         if (defined($extype1) and $op1_dtype ne $extype1) {
             # do casting
             $bdy .= "($extype1)(\@op1\@));";
         } else {
             $bdy .= "\@op1\@);";
         }
      }
           
   } else {
      die "Unknown unary operator is specified. ( = $op )\n";
   }
   if( !defined($bdy) ) { die "Fatal: $ari_dtype is not defined in \$operator_table.\n"}

   $_ = $bdy;
   s/^\s*/$idt/gm;
   s/\@op1\@/$op1/g;
   s/\@op2\@/$op2/g;
   if ( $ufunc_op==1 ) {
       s/\@op3\@/$op2/g;
   }
   s/\@ari_dtype\@/$ari_dtype/g;
   if ( $ufunc_op!=1 ) {
      if (/(\@cast_Bint1\@)(\(\s*[^,)]*\)+?)/) {
         my $cast = $1;
         my $arg  = $2;
         $_ = replace_cast_Bint($_,$op1_dtag,$cast,$arg);
      }
      if (/(\@ISNAN1\@|\@ISINF1\@)(\(\s*[^,)]*\)+?)/) {
         my ($ni, $f, $o);
         $f = $1;
         $o = $2;
         $_ = replace_isnan_or_isinf($_,$op1_dtag,$f,$o);
      }
   } else {
      if (/(\@cast_Bint2\@)(\(\s*[^,)]*\)+?)/) {
         my $cast = $1;
         my $arg  = $2;
         $_ = replace_cast_Bint($_,$op1_dtag,$cast,$arg);
      }
      if (/(\@cast_Bint1\@)(\(\s*[^,)]*\)+?)/) {
         my $cast = $1;
         my $arg  = $2;
         $_ = replace_cast_Bint($_,$op1_dtag,$cast,$arg);
      }
      if (/(\@ISNAN2\@|\@ISINF2\@)(\(\s*[^,)]*\)+?)/) {
         my ($ni, $f, $o);
         $f = $1;
         $o = $2;
         $_ = replace_isnan_or_isinf($_,$op1_dtag,$f,$o);
      }
      if (/(\@ISNAN1\@|\@ISINF1\@)(\(\s*[^,)]*\)+?)/) {
         my ($ni, $f, $o);
         $f = $1;
         $o = $2;
         $_ = replace_isnan_or_isinf($_,$op2_dtag,$f,$o);
      }
      if (/(\@COMPARE\@\(\s*([^,)]*),\s*([^,)]*),\s*([^,)]*)\))/) {
         my $e;
         $e =replace_compare($2,$op2_dtag,$3,$4,$op1_dtag);
         s/\Q$1\E/$e/g;
      }
   }
   chomp();
   $_ .= "\n";
   return $_;
}
################################ Binary Operator ###############################
sub binary_operator{
   my $op      = $_[0];
   my $ufunc_op= $_[1];
   my $idt     = $_[2];
   my $op1     = $_[3];
   my $op1_dtag= $_[4];
   my $op2     = $_[5];
   my $op2_dtag= $_[6];
   my $op3     = $_[7];
   my $op3_dtag= $_[8];
   my $ari_dtag= $_[9];
   my ($fn,$bdy,$op1_dtype,$op2_dtype,$op3_dtype,$ari_dtype,$extype1,$extype2);

   $op1_dtype = tag_to_Dtype($op1_dtag);
   $op2_dtype = tag_to_Dtype($op2_dtag);
   $op3_dtype = tag_to_Dtype($op3_dtag);
   $ari_dtype = tag_to_Atype($ari_dtag);
   #
   $fn = get_binary_intrinsic_name($op,$type1,$type2,$ari_dtype);
   if (exists($operator_table{$op})) {
      my @arg = ($op,$ari_dtype,$op1_dtype,$op2_dtype);
      $bdy = ref_operator_table(\%operator_table,\@arg);

   } elsif( defined($fn) ) {
      $extype1 = ${$c_intrinsic_types_table{$fn}}[0];
      $extype2 = ${$c_intrinsic_types_table{$fn}}[1];
      $bdy = "\@op3\@ = $fn(";
      if (defined($extype1) and $op1_dtype ne $extype1) {
          # do casting
          $bdy .= "($extype1)(\@op1\@),";
      } else {
          $bdy .= "\@op1\@,";
      }
      if (defined($extype2) and $op2_dtype ne $extype2) {
          # do casting
          $bdy .= "($extype2)(\@op2\@));";
      } else {
          $bdy .= "\@op2\@);";
      }
       
   } else {
      die "Unknown binary operator is specified. ( = $op )\n";
   }
   if( !defined($bdy) ) { die "Fatal: $ari_dtype is not defined in \$operator_table.\n"}

   $_ = $bdy;
   s/^\s*/$idt/gm;
   s/\@op1\@/$op1/g;
   s/\@op2\@/$op2/g;
   s/\@op3\@/$op3/g;
   s/\@ari_dtype\@/$ari_dtype/g;
   if (/(\@cast_Bint1\@)(\(\s*[^,)]*\)+?)/) {
      my $cast = $1;
      my $arg  = $2;
      $_ = replace_cast_Bint($_,$op1_dtag,$cast,$arg);
   }
   if (/(\@cast_Bint2\@)(\(\s*[^,)]*\)+?)/) {
      my $cast = $1;
      my $arg  = $2;
      $_ = replace_cast_Bint($_,$op2_dtag,$cast,$arg);
   }
   if (/(\@ISNAN1\@|\@ISINF1\@)(\(\s*[^,)]*\))/) {
      my ($ni, $f, $o);
      $f = $1;
      $o = $2;
      $_ = replace_isnan_or_isinf($_,$op1_dtag,$f,$o);
   }
   if (/(\@ISNAN2\@|\@ISINF2\@)(\(\s*[^,)]*\))/) {
      my ($ni, $f, $o);
      $f = $1;
      $o = $2;
      $_ = replace_isnan_or_isinf($_,$op2_dtag,$f,$o);
   }
   if (/(\@COMPARE\@\(\s*([^,)]*),\s*([^,)]*),\s*([^,)]*)\))/) {
      my $e;
      $e =replace_compare($2,$op1_dtag,$3,$4,$op2_dtag);
      s/\Q$1\E/$e/g;
   }
   chomp();
   $_ .= "\n";
}
############################ refer to the operator table #################################
sub ref_operator_table{
   my %table = %{$_[0]};
   my @arg   = @{$_[1]};
   my $gen   = $_[2] if $#_ == 2;
   my $a = shift(@arg);
   my $val;
   if (exists($table{$a})) {
      if (ref($table{$a})) {
         $gen = $table{"others"} if exists($table{"others"});
         $val = ref_operator_table($table{$a},\@arg, $gen);
      } else {
         $val = $table{$a};
      }
   } elsif (exists($table{"others"})) {
      if (ref($table{"others"})) {
         $val = ref_operator_table($table{"others"},\@arg,$gen);
      } else {
         $val = $table{"others"};
      }
   } elsif (defined($gen) and $gen ne '') {
         $val = $gen;
   } else {
         die "not defined in \$operator_table. ( = $a, @arg )\n";
   }
   return $val;
}
############################ Get Intrinsic Function Name ####################################
sub get_unary_intrinsic_name {
   my $op        = $_[0];
   my $op1_dtype = $_[1];
   my $ari_dtype = $_[2];
   my $fn;
   if (exists($irregular_intrinsic_table{$op})) {
      if (ref($irregular_intrinsic_table{$op})) {
         $fn = $irregular_intrinsic_table{$op}{$ari_dtype};
         if (exists($irregular_intrinsic_table{$op}{$ari_dtype})) {
            $fn = $irregular_intrinsic_table{$op}{$ari_dtype};
         } elsif (exists($irregular_intrinsic_table{$op}{"others"})) {
            $fn = $irregular_intrinsic_table{$op}{"others"};
         } else {
            die "not defined in \$irregular_intrinsic_table. ( = $fn, $ari_dtype )\n";
         }
      } else {
         $fn = $irregular_intrinsic_table{$op};
         chomp($fn);
         $fn = "$fn" . "f"       if $ari_dtype eq "float";
         $fn = "c" . "$fn"       if $ari_dtype eq "double _Complex";
         $fn = "c" . "$fn" . "f" if $ari_dtype eq "float _Complex";
      }
   } else {
      $fn = $op;
      $fn = "$fn" . "f"       if $ari_dtype eq "float";
      $fn = "c" . "$fn"       if $ari_dtype eq "double _Complex";
      $fn = "c" . "$fn" . "f" if $ari_dtype eq "float _Complex";
   }
   return $fn;
} 

sub get_binary_intrinsic_name {
   my $op        = $_[0];
   my $op1_dtype = $_[1];
   my $op2_dtype = $_[2];
   my $ari_dtype = $_[3];
   my $fn;
   # change the output datatype corresponding to input datatypes
#   if      ($op1_dtype eq "double _Complex" or $op2_dtype eq "double _Complex") { $ari_dtype = "double _Complex";
#   } elsif ($op1_dtype eq "float _Complex"  or $op2_dtype eq "float _Complex" ) { $ari_dtype = "float _Complex";
#   } elsif ($op1_dtype eq "double"          or $op2_dtype eq "double"         ) { $ari_dtype = "double";
#   } elsif ($op1_dtype eq "float"           or $op2_dtype eq "float"          ) { $ari_dtype = "float";
#   }
   if (exists($irregular_intrinsic_table{$op})) {
      if (ref($irregular_intrinsic_table{$op})) {
         $fn = $irregular_intrinsic_table{$op}{$ari_dtype};
      } else {
         $fn = $irregular_intrinsic_table{$op};
         chomp($fn);
         $fn = "$fn" . "f"       if $ari_dtype eq "float";
         $fn = "c" . "$fn"       if $ari_dtype eq "double _Complex";
         $fn = "c" . "$fn" . "f" if $ari_dtype eq "float _Complex";
      }
   } else {
      $fn = $op;
      $fn = "$fn" . "f"       if $ari_dtype eq "float";
      $fn = "c" . "$fn"       if $ari_dtype eq "double _Complex";
      $fn = "c" . "$fn" . "f" if $ari_dtype eq "float _Complex";
   }
   return $fn;
} 

sub tag_to_Dtype {
   my $dtag  = $_[0];
   my $dtype;
   if      ($dtag eq "c128") { $dtype="double _Complex";
   } elsif ($dtag eq "c64")  { $dtype="float _Complex";
   } elsif ($dtag eq "f64")  { $dtype="double";
   } elsif ($dtag eq "f32")  { $dtype="float";
   } elsif ($dtag eq "i64")  { $dtype="int64_t";
   } elsif ($dtag eq "i32")  { $dtype="int32_t";
   } elsif ($dtag eq "u64")  { $dtype="uint64_t";
   } elsif ($dtag eq "u32")  { $dtype="uint32_t";
   } elsif ($dtag eq "bool") { $dtype="int32_t";
   } else {
      die "an undefined data tag is detected. ( = $op, $dtag )\n";
   }
   return $dtype;
}

sub tag_to_Atype {
   my $atag  = $_[0];
   my $atype;
   if      ($atag eq "c128") { $atype="double _Complex";
   } elsif ($atag eq "c64")  { $atype="float _Complex";
   } elsif ($atag eq "f64")  { $atype="double";
   } elsif ($atag eq "f32")  { $atype="float";
   } elsif ($atag eq "i64")  { $atype="int64_t";
   } elsif ($atag eq "i32")  { $atype="int32_t";
   } elsif ($atag eq "u64")  { $atype="uint64_t";
   } elsif ($atag eq "u32")  { $atype="uint32_t";
   } elsif ($atag eq "bool") { $atype="bool";
   } else {
      die "an undefined data tag is detected. ( = $op, $atag )\n";
   }
   return $atype;
}

sub replace_cast_Bint{
   $_    = $_[0];
   my $dtag = $_[1];
   my $cast = $_[2];
   my $arg  = $_[3];
   if     ($dtag eq "c128") { s/$cast\Q$arg\E/(f64_to_Bint(creal$arg)||f64_to_Bint(cimag$arg))/g;
   }elsif ($dtag eq "c64")  { s/$cast\Q$arg\E/(f32_to_Bint(crealf$arg)||f32_to_Bint(cimagf$arg))/g;
   }elsif ($dtag eq "f64")  { s/$cast\Q$arg\E/f64_to_Bint$arg/g;
   }elsif ($dtag eq "f32")  { s/$cast\Q$arg\E/f32_to_Bint$arg/g;
   }elsif ($dtag eq "u64")  { s/$cast\Q$arg\E/u64_to_Bint$arg/g;
   }elsif ($dtag eq "u32")  { s/$cast\Q$arg\E/u32_to_Bint$arg/g;
   }elsif ($dtag eq "i64")  { s/$cast\Q$arg\E/i64_to_Bint$arg/g;
   }elsif ($dtag eq "i32")  { s/$cast\Q$arg\E/i32_to_Bint$arg/g;
   }elsif ($dtag eq "bool") { s/$cast\Q$arg\E/$arg/g;
   } else {
      die "Unknown data type is specified. ( = $dtag )\n";
   }
   return $_;
}

sub replace_isnan_or_isinf{
   $_    = $_[0];
   my $dtag = $_[1];
   my $f    = $_[2];
   my $arg  = $_[3];
   my $ni;
   if ($f =~ /\@ISNAN/ ) { $ni ="isnan";
   } else                { $ni ="isinf";
   }
   if      ($dtag eq "c128") { s/$f\Q$arg\E/${ni}_f64(creal$arg)||${ni}_f64(cimag$arg)/g;
   } elsif ($dtag eq "c64")  { s/$f\Q$arg\E/${ni}_f32(crealf$arg)||${ni}_f32(cimagf$arg)/g;
   } elsif ($dtag eq "f64")  { s/$f\Q$arg\E/${ni}_f64$arg/g;
   } elsif ($dtag eq "f32")  { s/$f\Q$arg\E/${ni}_f32$arg/g;
   } elsif ($dtag eq "u64")  { s/$f\Q$arg\E/0/g;
   } elsif ($dtag eq "u32")  { s/$f\Q$arg\E/0/g;
   } elsif ($dtag eq "i64")  { s/$f\Q$arg\E/0/g;
   } elsif ($dtag eq "i32")  { s/$f\Q$arg\E/0/g;
   } elsif ($dtag eq "bool") { s/$f\Q$arg\E/0/g;
   } else {
      die "Unknown data type is specified. ( = $dtag )\n";
   }
   return $_;
}

sub replace_compare{
   my $op1   = $_[0];
   my $dtag1 = $_[1];
   my $op    = $_[2];
   my $op2   = $_[3];
   my $dtag2 = $_[4];
   my ($exp, $r1, $i1, $r2, $i2, $uint1, $uint2);
   if      ($dtag1 eq "c128") {$r1 = "creal($op1)";  $i1 = "cimag($op1)";
   } elsif ($dtag1 eq "c64")  {$r1 = "crealf($op1)"; $i1 = "cimagf($op1)";
   } else                     {$r1 = "$op1";         $i1 = "0";
   }
   if      ($dtag2 eq "c128") {$r2 = "creal($op2)";  $i2 = "cimag($op2)";
   } elsif ($dtag2 eq "c64")  {$r2 = "crealf($op2)"; $i2 = "cimagf($op2)";
   } else                     {$r2 = "$op2";         $i2 = "0";
   }
   if ($dtag1 eq "u64"  or $dtag1 eq "u32") { $uint1 = 1;
   } else                                   { $uint1 = 0;
   }
   if ($dtag2 eq "u64"  or $dtag2 eq "u32") { $uint2 = 1;
   } else                                   { $uint2 = 0;
   }
   if      ($dtag1 eq "c128" or $dtag2 eq "c128") { $exp = "($r1==$r2) ? ($i1 $op $i2) : ($r1 $op $r2)";
   } elsif ($dtag1 eq "c64"  or $dtag2 eq "c64")  { $exp = "($r1==$r2) ? ($i1 $op $i2) : ($r1 $op $r2)";
   } elsif ($uint1 eq 1     and $uint2 eq 0    )  { $exp = "($r2>=0  ) ? ($r1 $op $r2) : (0   $op $r2)";
   } elsif ($uint1 eq 0     and $uint2 eq 1    )  { $exp = "($r1>=0  ) ? ($r1 $op $r2) : ($r1 $op 0)";
   } else                                         { $exp = "$r1 $op $r2";
   }
   return $exp;
}

################################### Table ####################################
sub create_table {

my $i32 = "i32";
my $i64 = "i64";
my $u32 = "u32";
my $u64 = "u64";
my $f32 = "f32";
my $f64 = "f64";
my $c64 = "c64";
my $c128= "c128";
my $bool= "bool";
my $i32_t = "int32_t";
my $i64_t = "int64_t";
my $u32_t = "uint32_t";
my $u64_t = "uint64_t";
my $f32_t = "float";
my $f64_t = "double";
my $c64_t = "float _Complex";
my $c128_t= "double _Complex";
my $bool_t= "bool";
my $r2d = "57.29577951308232087679e0"; #180/pi
my $d2r = "1.7453292519943295769e\-2"; #pi/180

%dtag_table = (
# Math operatios
'cast'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'fill'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'add'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'subtract' => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'multiply' => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'divide' => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'logaddexp' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64", "$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'logaddexp2' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64", "$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'true_divide' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'floor_divide' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'negative'   => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'positive'   => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'power' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'remainder' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64", "$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'mod' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64", "$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'fmod' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64", "$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'divmod' => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64", "$i32, $i64, $u32, $u64, $f32, $f64"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64", "$i32, $i64, $u32, $u64, $f32, $f64"],
},
'absolute'=> {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'dtype'=> ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'fabs'=> {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$f32, $f64, $c64, $c128"],
},
'rint'    => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'sign' => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'heaviside'=> {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64", "$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'conj'    => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'conjugate'=> {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'exp'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'exp2'     => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$f32, $f64, $c64, $c128"],
},
'log'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'log2'    => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$f32, $f64, $c64, $c128"],
},
'log10'    => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$f32, $f64, $c64, $c128"],
},
'expm1'     => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$f32, $f64, $c64, $c128"],
},
'log1p'     => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$f32, $f64, $c64, $c128"],
},
'sqrt'    => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'square' => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'cbrt'    => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$f32, $f64, $c64, $c128"],
},
'reciprocal' => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},

# gcd and lcm are not inplemented now.
'gcd'=> {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64", "$i32, $i64, $u32, $u64, $f32, $f64"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64"],
},
'lcm'=> {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64", "$i32, $i64, $u32, $u64, $f32, $f64"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64"],
},
# Trigonometric functions
'sin'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'cos'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'tan'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'arcsin'  => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'arccos'  => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'arctan'  => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'arctan2' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64", "$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'hypot'    => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'dtype'=> ["$f32, $f64, $c64, $c128"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'sinh'    => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'cosh'    => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'tanh'    => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'arcsinh' => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'arccosh' => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'arctanh' => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $c64, $c128"],
},
'deg2rad' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'rad2deg' => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'degrees' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'radians' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
# Bit-twiddling functions
'bitwise_and' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64", "$bool, $i32, $i64, $u32, $u64"],
    'dtype'=> ["$bool, $i32, $i64, $u32, $u64"],
    'out'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'bitwise_or' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64", "$bool, $i32, $i64, $u32, $u64"],
    'dtype'=> ["$bool, $i32, $i64, $u32, $u64"],
    'out'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'bitwise_xor' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64", "$bool, $i32, $i64, $u32, $u64"],
    'dtype'=> ["$bool, $i32, $i64, $u32, $u64"],
    'out'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'invert'   => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64"],
    'dtype'=> ["$bool, $i32, $i64, $u32, $u64"],
    'out'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'left_shift' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64", "$bool, $i32, $i64, $u32, $u64"],
    'dtype'=> ["$bool, $i32, $i64, $u32, $u64"],
    'out'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'right_shift' => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64", "$bool, $i32, $i64, $u32, $u64"],
    'dtype'=> ["$bool, $i32, $i64, $u32, $u64"],
    'out'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'greater' => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'greater_equal' => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'less' => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'less_equal' => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'equal'   => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'not_equal' => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'logical_and' => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'logical_or' => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'logical_xor' => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'logical_not' => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'maximum' => { 
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'minimum' => { 
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'fmax' => { 
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'fmin' => { 
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'argmax' => { 
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$i64"],
    'out' => ["$i64"],
},
'argmin' => { 
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$i64"],
    'out' => ["$i64"],
},
# Floating functions
'isfinite'   => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'isinf'   => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'isnan'   => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'isnat'   => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'signbit'   => {
    'in'   => ["$i32, $i64, $u32, $u64, $f32, $f64, $bool"],
    'dtype'=> ["$bool"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'copysign'   => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64", "$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'nextafter'=> { 
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64", "$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'spacing'   => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$f32, $f64, $c64, $c128"],
},
'modf'=> { 
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$f32, $f64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'ldexp'=> { 
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64", "$bool, $i32, $i64, $u32, $u64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$f32, $f64, $c64, $c128"],
},
'frexp'=> { 
    'in'   => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'dtype'=> ["$f32, $f64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out'  => ["$f32, $f64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'floor'   => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'ceil'    => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
'trunc'   => {
    'in'   => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64"],
    'dtype'=> ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
},
# Other mathematical functions
#'real'     => {
#    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
#    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $bool"],
#},
#'imag'     => {
#    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
#    'out' => ["$i32, $i64, $u32, $u64, $f32, $f64, $bool"],
#},
'angle'     => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128, $bool"],
    'out' => ["$f32, $f64"],
},
# Special functions
'erf'     => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out' => ["$f32, $f64"],
},
'erfc'    => {
    'in'  => ["$i32, $i64, $u32, $u64, $f32, $f64"],
    'out' => ["$f32, $f64"],
},
# Creation
'copy'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'copy_masked'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
'arange'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
# Indexing


# Linear Algebra
'dot'     => {
    'in'  => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128", "$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
    'out' => ["$bool, $i32, $i64, $u32, $u64, $f32, $f64, $c64, $c128"],
},
);

%operator_table = (
'negative'   => '@op2@ = - (@ari_dtype@)@op1@;',
'positive'   => '@op2@ = @op1@;',
'invert'     => {
     'others'          => '@op2@ = ~((@ari_dtype@)@op1@);',
     'bool'            => '@op2@ = (@op1@) ? 0 : 1;',
},
'logical_not'=> '@op2@ = (@ari_dtype@)(!@cast_Bint1@(@op1@));',
'reciprocal' => '@op2@ = (@ari_dtype@)((@ari_dtype@)1 / @op1@);',
'square'     => '@op2@ = (@ari_dtype@)(@op1@ * @op1@);',
'hypot'=> {
     'float _Complex'  => '@op3@ = csqrtf((float _Complex)(@op1@)*(@op1@) + (float _Complex)(@op2@)*(@op2@));',
     'double _Complex' => '@op3@ = csqrt ((double _Complex)(@op1@)*(@op1@) + (double _Complex)(@op2@)*(@op2@));',
     'float'           => '@op3@ =  hypotf((float)(@op1@), (float)(@op2@));',
     'double'          => '@op3@ =  hypot ((double)(@op1@), (double)(@op2@));',
},
'degrees'    => '@op2@ = (@op1@) * (@ari_dtype@)57.29577951308232087679e0;',
'rad2deg'    => '@op2@ = (@op1@) * (@ari_dtype@)57.29577951308232087679e0;',
'deg2rad'    => '@op2@ = (@op1@) * (@ari_dtype@)1.7453292519943295769e-02;',
'radians'    => '@op2@ = (@op1@) * (@ari_dtype@)1.7453292519943295769e-02;',
'sign'=> { 
     'float _Complex'  => '@op2@ = (crealf(@op1@)>0) ? 1 : ( (crealf(@op1@)<0) ? -1 : ((crealf(@op1@)==0) ? 0 : crealf(@op1@)) );',
     'double _Complex' => '@op2@ = (creal (@op1@)>0) ? 1 : ( (creal (@op1@)<0) ? -1 : ((creal (@op1@)==0) ? 0 : creal (@op1@)) );',
     'float'           => '@op2@ = ((@ari_dtype@)(@op1@)>0) ? 1 : ( ((@ari_dtype@)(@op1@)<0) ? -1 : (((@ari_dtype@)(@op1@)==0) ? 0 : @op1@) );',
     'double'          => '@op2@ = ((@ari_dtype@)(@op1@)>0) ? 1 : ( ((@ari_dtype@)(@op1@)<0) ? -1 : (((@ari_dtype@)(@op1@)==0) ? 0 : @op1@) );',
     'int32_t'         => '@op2@ = ((@ari_dtype@)(@op1@)>0) ? 1 : ( ((@ari_dtype@)(@op1@)<0) ? -1 : (((@ari_dtype@)(@op1@)==0) ? 0 : @op1@) );',
     'int64_t'         => '@op2@ = ((@ari_dtype@)(@op1@)>0) ? 1 : ( ((@ari_dtype@)(@op1@)<0) ? -1 : (((@ari_dtype@)(@op1@)==0) ? 0 : @op1@) );',
     'uint32_t'        => '@op2@ = ((@ari_dtype@)(@op1@)==0) ? 0 : 1;',
     'uint64_t'        => '@op2@ = ((@ari_dtype@)(@op1@)==0) ? 0 : 1;',
},
'cast'=> { 
     'others'          => '@op2@ = (@ari_dtype@)(@op1@);',
     'bool'            => '@op2@ = @cast_Bint1@(@op1@);',
},
'add'=> { 
     'others'          => '@op3@ = (@ari_dtype@)(@op1@) + (@ari_dtype@)(@op2@);',
     'bool'            => '@op3@ = @op1@ || @op2@;',
},
'subtract'=> { 
     'others'          => '@op3@ = (@ari_dtype@)(@op1@) - (@ari_dtype@)(@op2@);',
     'bool'            => '@op3@ = @op1@ && !@op2@ || !@op1@ && @op2@;',
},
'multiply'    => { 
     'others'          => '@op3@ = (@ari_dtype@)(@op1@) * (@ari_dtype@)(@op2@);',
     'bool'            => '@op3@ = @op1@ && @op2@;',
},
'divide'      => '@op3@ = (@ari_dtype@)(@op1@) / (@ari_dtype@)(@op2@);',
'logaddexp'=> {
     'others'          => '@op3@ = (@op1@>@op2@) ? (@ISINF1@(@op2@)) ? @op1@ : @op1@+log1p(exp((double)@op2@-(double)@op1@)) : (@ISINF2@(@op1@)) ? @op2@ : @op2@+log1p(exp((double)@op1@-(double)@op2@));',
},
'logaddexp2'=> {
     'others'          => '@op3@ = (@op1@>@op2@) ? (@ISINF1@(@op2@)) ? @op1@ : @op1@+log1p(pow((double)2.0,(double)@op2@-(double)@op1@))/(double)0.693147180559945286 : (@ISINF2@(@op1@)) ? @op2@ : @op2@+log1p(pow((double)2.0,(double)@op1@-(double)@op2@))/(double)0.693147180559945286;',
},
'true_divide' => '@op3@ = (@ari_dtype@)(@op1@) / (@ari_dtype@)(@op2@);',
'floor_divide'=> {
     'float'           => '@op3@ = floorf((float)(@op1@)/(float)(@op2@)); if ((float)@op2@==0) @op3@ = nanf("n");',
     'double'          => '@op3@ = floor ((double)(@op1@)/(double)(@op2@)); if ((double)@op2@==0) @op3@ = nan("n");',
     'float _Complex'  => '@op3@ = floorf(crealf((float _Complex)(@op1@)/(float _Complex)(@op2@))); if (cabsf((float _Complex)@op2@)==0) @op3@ = nanf("n");',
     'double _Complex' => '@op3@ = floor (creal((double _Complex)(@op1@)/(double _Complex)(@op2@))); if (cabsf((double _Complex)@op2@)==0) @op3@ = nan("n");',
     'others'          => << 'EOS'
                          const @ari_dtype@ l = (@op2@!=0) ? ((@ari_dtype@)@op1@ / (@ari_dtype@)@op2@) * (@ari_dtype@)@op2@ : @op1@;
                          @op3@ = (( ((double)((@ari_dtype@)@op1@) * (double)@op2@)>=0 || l == @op1@) ? l : l - (@ari_dtype@)@op2@)/(@ari_dtype@)(@op2@);
EOS
,
},
'power'       => { 
     'float'           => '@op3@ = powf ((@ari_dtype@)(@op1@), (@ari_dtype@)(@op2@));',
     'double'          => '@op3@ = pow  ((@ari_dtype@)(@op1@), (@ari_dtype@)(@op2@));',
     'float _Complex'  => '@op3@ = (crealf((float _Complex)@op1@)==0&&cimagf((float _Complex)@op1@)==0&&cimagf((float _Complex)@op2@)!=0) ? nanf("n")+nanf("n")*I : cpowf((float _Complex)@op1@, (float _Complex)@op2@);',
     'double _Complex' => '@op3@ = (creal((double _Complex)@op1@)==0&&cimag((double _Complex)@op1@)==0&&cimag((double _Complex)@op2@)!=0) ? nan("n")+nan("n")*I : cpow((double _Complex)@op1@, (double _Complex)@op2@);',
     'others'          => << 'EOS'
                  double t = pow((double)(@op1@), (double)(@op2@));
                  @op3@ = (t>=0) ? (@ari_dtype@) (t+0.5) : (@ari_dtype@) (t-0.5);
EOS
,
},
'mod'        => { 
     'float'           => '@op3@ = (float)(@op1@) - floorf((float)(@op1@) / (float)(@op2@)) * (float)@op2@;',
     'double'          => '@op3@ = (double)(@op1@) - floor((double)(@op1@) / (double)(@op2@)) * (double)@op2@;',
     'others'          => << 'EOS'
                          const @ari_dtype@ l = (@op2@!=0) ? ((@ari_dtype@)@op1@ / (@ari_dtype@)@op2@) * (@ari_dtype@)@op2@ : @op1@;
                          @op3@ = @op1@ - (( ((double)@op1@ * (double)@op2@)>=0 || l == @op1@) ? l : l - @op2@);
EOS
,
},
'remainder'        => { 
     'float'           => '@op3@ = (float)(@op1@) - floorf((float)(@op1@) / (float)(@op2@)) * (float)@op2@;',
     'double'          => '@op3@ = (double)(@op1@) - floor((double)(@op1@) / (double)(@op2@)) * (double)@op2@;',
     'others'          => << 'EOS'
                          const @ari_dtype@ l = (@op2@!=0) ? ((@ari_dtype@)@op1@ / (@ari_dtype@)@op2@) * (@ari_dtype@)@op2@ : @op1@;
                          @op3@ = @op1@ - (( ((double)@op1@ * (double)@op2@)>=0 || l == @op1@) ? l : l - @op2@);
EOS
,
},
'fmod'       => { 
     'float'           => '@op3@ = fmodf((float)(@op1@),(float)(@op2@));',
     'double'          => '@op3@ = fmod((double)(@op1@),(double)(@op2@));',
     'others'          => '@op3@ = (@op2@) ? (@ari_dtype@)(@op1@)%(@ari_dtype@)(@op2@) : 0;',
},
'spacing'       => { 
     'float'           => '@op2@ = (@op1@>0) ? nextafterf((float)@op1@,(float)FLT_MAX)-@op1@ : nextafterf((float)@op1@,-(float)FLT_MAX)-@op1@;',
     'double'          => '@op2@ = (@op1@>0) ? nextafter((double)@op1@,(double)FLT_MAX)-@op1@ : nextafter((double)@op1@,-(double)FLT_MAX)-@op1@;',
},
'bitwise_and'  => { 
     'others'          => '@op3@ = (@ari_dtype@)(@op1@) & (@ari_dtype@)(@op2@);',
     'bool'            => '@op3@ = @cast_Bint1@(@op1@) & @cast_Bint2@(@op2@);',
},
'bitwise_xor'  => { 
     'others'          => '@op3@ = (@ari_dtype@)(@op1@) ^ (@ari_dtype@)(@op2@);',
     'bool'            => '@op3@ = @cast_Bint1@(@op1@) ^ @cast_Bint2@(@op2@);',
},
'bitwise_or'   => { 
     'others'          => '@op3@ = (@ari_dtype@)(@op1@) | (@ari_dtype@)(@op2@);',
     'bool'            => '@op3@ = @cast_Bint1@(@op1@) | @cast_Bint2@(@op2@);',
},
'logical_and'  => '@op3@ = (@cast_Bint1@(@op1@) && @cast_Bint2@(@op2@));',
'logical_or'   => '@op3@ = (@cast_Bint1@(@op1@) || @cast_Bint2@(@op2@));',
'logical_xor'  => '@op3@ = (@cast_Bint1@(@op1@) != @cast_Bint2@(@op2@));',
'left_shift'=> { 
     'bool'            => '@op3@ = (@op2@>31) ? 0 : (@ari_dtype@)(@op1@) << (@ari_dtype@)(@op2@);',
     'int32_t'         => '@op3@ = (@op2@>31) ? 0 : (@ari_dtype@)(@op1@) << (@ari_dtype@)(@op2@);',
     'int64_t'         => '@op3@ = (@op2@>63) ? 0 : (@ari_dtype@)(@op1@) << (@ari_dtype@)(@op2@);',
     'uint32_t'        => '@op3@ = (@op2@>31) ? 0 : (@ari_dtype@)(@op1@) << (@ari_dtype@)(@op2@);',
     'uint64_t'        => '@op3@ = (@op2@>63) ? 0 : (@ari_dtype@)(@op1@) << (@ari_dtype@)(@op2@);',
},
'right_shift'=> { 
     'bool'            => '@op3@ = (@op2@>31) ? 0 : (@ari_dtype@)(@op1@) >> (@ari_dtype@)(@op2@);',
     'int32_t'         => '@op3@ = (@op2@>31) ? 0 : (@ari_dtype@)(@op1@) >> (@ari_dtype@)(@op2@);',
     'int64_t'         => '@op3@ = (@op2@>63) ? 0 : (@ari_dtype@)(@op1@) >> (@ari_dtype@)(@op2@);',
     'uint32_t'        => '@op3@ = (@op2@>31) ? 0 : (@ari_dtype@)(@op1@) >> (@ari_dtype@)(@op2@);',
     'uint64_t'        => '@op3@ = (@op2@>63) ? 0 : (@ari_dtype@)(@op1@) >> (@ari_dtype@)(@op2@);',
},
'less'         => '@op3@ = (@COMPARE@(@op1@,<,@op2@)) ? 1 : 0;',
'less_equal'   => '@op3@ = (@COMPARE@(@op1@,<=,@op2@)) ? 1 : 0;',
'greater'      => '@op3@ = (@COMPARE@(@op1@,>,@op2@)) ? 1 : 0;',
'greater_equal'=> '@op3@ = (@COMPARE@(@op1@,>=,@op2@)) ? 1 : 0;',
'equal'        => '@op3@ = (@COMPARE@(@op1@,==,@op2@)) ? 1 : 0;',
'not_equal'    => '@op3@ = (@COMPARE@(@op1@,!=,@op2@)) ? 1 : 0;',
'argmax'      => {
     'others'  => << 'EOS'
              const Bint b1 = @ISNAN1@(@op1@);
              const Bint b2 = @ISNAN2@(@op2@);
              if      (b1) @op3@ = 1;
              else if (b2) @op3@ = 0;
              else         @op3@ = @COMPARE@(@op1@,>,@op2@);
EOS
,
     'bool'    => '@op3@ = @op1@ && !@op2@;',
},
'argmin'      => {
     'others'  => << 'EOS'
              const Bint b1 = @ISNAN1@(@op1@);
              const Bint b2 = @ISNAN2@(@op2@);
              if      (b1) @op3@ = 1;
              else if (b2) @op3@ = 0;
              else         @op3@ = @COMPARE@(@op1@,<,@op2@);
EOS
,
     'bool'    => '@op3@ = !@op1@ && @op2@;',
},
'maximum'      => {
     'others'  => << 'EOS'
              const Bint b1 = @ISNAN1@(@op1@);
              const Bint b2 = @ISNAN2@(@op2@);
              if      (b1) @op3@ = (@ari_dtype@)@op1@;
              else if (b2) @op3@ = (@ari_dtype@)@op2@;
              else         @op3@ = (@COMPARE@(@op1@,>,@op2@)) ? (@ari_dtype@)@op1@ : (@ari_dtype@)@op2@;
EOS
,
     'bool'    => '@op3@ = @op1@ || @op2@;',
},
'minimum'      => {
     'others'  => << 'EOS'
              const Bint b1 = @ISNAN1@(@op1@);
              const Bint b2 = @ISNAN2@(@op2@);
              if      (b1) @op3@ = (@ari_dtype@)@op1@;
              else if (b2) @op3@ = (@ari_dtype@)@op2@;
              else         @op3@ = (@COMPARE@(@op1@,<,@op2@)) ? (@ari_dtype@)@op1@ : (@ari_dtype@)@op2@;
EOS
,
     'bool'    => '@op3@ = @op1@ && @op2@;',
},
'fmax'      => {
     'others'  => << 'EOS'
              const Bint b1 = @ISNAN1@(@op1@);
              const Bint b2 = @ISNAN2@(@op2@);
              if      (b2) @op3@ = @op1@;
              else if (b1) @op3@ = @op2@;
              else         @op3@ = (@COMPARE@(@op1@,>,@op2@)) ? (@ari_dtype@)@op1@ : (@ari_dtype@)@op2@;
EOS
,
     'bool'    => '@op3@ = @op1@ || @op2@;',
},
'fmin'      => {
     'others'  => << 'EOS'
              const Bint b1 = @ISNAN1@(@op1@);
              const Bint b2 = @ISNAN2@(@op2@);
              if      (b2) @op3@ = @op1@;
              else if (b1) @op3@ = @op2@;
              else         @op3@ = (@COMPARE@(@op1@,<,@op2@)) ? (@ari_dtype@)@op1@ : (@ari_dtype@)@op2@;
EOS
,
     'bool'    => '@op3@ = @op1@ && @op2@;',
},
'heaviside'=> { 
     'float'           => '@op3@ = ((@ari_dtype@)(@op1@)<0) ? 0 : (((@ari_dtype@)(@op1@==0)) ? @op2@ : 1);',
     'double'          => '@op3@ = ((@ari_dtype@)(@op1@)<0) ? 0 : (((@ari_dtype@)(@op1@==0)) ? @op2@ : 1);',
     'int32_t'         => '@op3@ = ((@ari_dtype@)(@op1@)<0) ? 0 : (((@ari_dtype@)(@op1@==0)) ? @op2@ : 1);',
     'int64_t'         => '@op3@ = ((@ari_dtype@)(@op1@)<0) ? 0 : (((@ari_dtype@)(@op1@==0)) ? @op2@ : 1);',
     'uint32_t'        => '@op3@ = ((@ari_dtype@)(@op1@==0)) ? @op2@ : 1;',
     'uint64_t'        => '@op3@ = ((@ari_dtype@)(@op1@==0)) ? @op2@ : 1;',
},
'dot'         => '@op3@ += (@ari_dtype@)(@op1@) * (@ari_dtype@)(@op2@);',
'copy'=> { 
     'others'          => '@op2@ = (@ari_dtype@)(@op1@);',
     'bool'            => '@op2@ = @cast_Bint1@(@op1@);',
},
'copy_masked'=> { 
     'others'          => '@op2@ = (@ari_dtype@)(@op1@);',
     'bool'            => '@op2@ = @cast_Bint1@(@op1@);',
},
'isinf'=> { 
     'others'          => '@op2@ = @ISINF1@(@op1@);',
},
'isnan'=> { 
     'others'          => '@op2@ = @ISNAN1@(@op1@);',
},
'arccos'       => { 
     'float _Complex'  => '@op2@ = cacosf((float _Complex)@op1@);',
     'double _Complex' => '@op2@ = cacos ((double _Complex)@op1@);',
     'float'           => '@op2@ =  acosf((float)(@op1@));',
     'others'          => '@op2@ =  acos ((double)(@op1@));',
},
'arctanh'       => { 
     'float _Complex'  => '@op2@ = catanhf((float _Complex)@op1@);',
     'double _Complex' => '@op2@ = catanh ((double _Complex)@op1@);',
     'float'           => '@op2@ =  atanhf((float)(@op1@));',
     'others'          => '@op2@ =  atanh ((double)(@op1@));',
},
'ldexp'       => { 
     'float'  => {
         # operand 1
         'uint32_t'  => {
             # operand 2
             'int32_t'          => '@op3@=ldexpf((float)@op1@,@op2@);',
             'int64_t'          => '@op3@=(@op2@<INT32_MIN) ? 0 : ((@op2@>INT32_MAX) ? ( @op1@ == 0 ? 0 : FLT_MAX * @op2@) : ldexpf((float)@op1@,(int32_t)@op2@));',
             'uint32_t'         => '@op3@=                        ((@op2@>INT32_MAX) ? ( @op1@ == 0 ? 0 : FLT_MAX * @op2@) : ldexpf((float)@op1@,(int32_t)@op2@));',
             'uint64_t'         => '@op3@=                        ((@op2@>INT32_MAX) ? ( @op1@ == 0 ? 0 : FLT_MAX * @op2@) : ldexpf((float)@op1@,(int32_t)@op2@));',
             'bool'             => '@op3@=(@op2@ ? @op1@ * 2.0 : @op1@)'
         },
         'uint64_t'  => { # same as uint32_t
             # operand 2
             'int32_t'          => '@op3@=ldexpf((float)@op1@,@op2@);',
             'int64_t'          => '@op3@=(@op2@<INT32_MIN) ? 0 : ((@op2@>INT32_MAX) ? ( @op1@ == 0 ? 0 : FLT_MAX * @op2@) : ldexpf((float)@op1@,(int32_t)@op2@));',
             'uint32_t'         => '@op3@=                        ((@op2@>INT32_MAX) ? ( @op1@ == 0 ? 0 : FLT_MAX * @op2@) : ldexpf((float)@op1@,(int32_t)@op2@));',
             'uint64_t'         => '@op3@=                        ((@op2@>INT32_MAX) ? ( @op1@ == 0 ? 0 : FLT_MAX * @op2@) : ldexpf((float)@op1@,(int32_t)@op2@));',
             'bool'             => '@op3@=(@op2@ ? @op1@ * 2.0 : @op1@)'
         },
         'others'  => {
             # operand 2
             'int32_t'          => '@op3@=ldexpf((float)@op1@,@op2@);',
             'int64_t'          => '@op3@=(@op2@<INT32_MIN) ? 0 : ((@op2@>INT32_MAX) ? ((@op1@ > 0) ? FLT_MAX * @op2@ : ((@op1@ < 0) ? - FLT_MAX * @op2@ : 0)): ldexpf((float)@op1@,(int32_t)@op2@));',
             'uint32_t'         => '@op3@=                        ((@op2@>INT32_MAX) ? ((@op1@ > 0) ? FLT_MAX * @op2@ : ((@op1@ < 0) ? - FLT_MAX * @op2@ : 0)): ldexpf((float)@op1@,(int32_t)@op2@));',
             'uint64_t'         => '@op3@=                        ((@op2@>INT32_MAX) ? ((@op1@ > 0) ? FLT_MAX * @op2@ : ((@op1@ < 0) ? - FLT_MAX * @op2@ : 0)): ldexpf((float)@op1@,(int32_t)@op2@));',
             'bool'             => '@op3@=(@op2@ ? @op1@ * 2.0 : @op1@)'
         },
         'bool'  => { # same as uint32_t
             # operand 2
             'int32_t'          => '@op3@=ldexpf((float)@op1@,@op2@);',
             'int64_t'          => '@op3@=(@op2@<INT32_MIN) ? 0 : ((@op2@>INT32_MAX) ? ( @op1@ == 0 ? 0 : FLT_MAX * @op2@) : ldexpf((float)@op1@,(int32_t)@op2@));',
             'uint32_t'         => '@op3@=                        ((@op2@>INT32_MAX) ? ( @op1@ == 0 ? 0 : FLT_MAX * @op2@) : ldexpf((float)@op1@,(int32_t)@op2@));',
             'uint64_t'         => '@op3@=                        ((@op2@>INT32_MAX) ? ( @op1@ == 0 ? 0 : FLT_MAX * @op2@) : ldexpf((float)@op1@,(int32_t)@op2@));',
             'bool'             => '@op3@=(@op2@ ? @op1@ * 2.0 : @op1@)'
         },
     },
     'double'  => {
         # operand 1
         'uint32_t'  => {
             # operand 2
             'int32_t'          => '@op3@=ldexp((double)@op1@,@op2@);',
             'int64_t'          => '@op3@=(@op2@<INT32_MIN) ? 0: ((@op2@>INT32_MAX) ? (@op1@ == 0 ? 0 : DBL_MAX * @op2@) : ldexp((double)@op1@,(int32_t)@op2@));',
             'uint32_t'         => '@op3@=                       ((@op2@>INT32_MAX) ? (@op1@ == 0 ? 0 : DBL_MAX * @op2@) : ldexp((double)@op1@,(int32_t)@op2@));',
             'uint64_t'         => '@op3@=                       ((@op2@>INT32_MAX) ? (@op1@ == 0 ? 0 : DBL_MAX * @op2@) : ldexp((double)@op1@,(int32_t)@op2@));',
             'bool'             => '@op3@=(@op2@ ? @op1@ * 2.0 : @op1@)'
         },
         'uint64_t'  => { # same as uint32_t
             # operand 2
             'int32_t'          => '@op3@=ldexp((double)@op1@,@op2@);',
             'int64_t'          => '@op3@=(@op2@<INT32_MIN) ? 0: ((@op2@>INT32_MAX) ? (@op1@ == 0 ? 0 : DBL_MAX * @op2@) : ldexp((double)@op1@,(int32_t)@op2@));',
             'uint32_t'         => '@op3@=                       ((@op2@>INT32_MAX) ? (@op1@ == 0 ? 0 : DBL_MAX * @op2@) : ldexp((double)@op1@,(int32_t)@op2@));',
             'uint64_t'         => '@op3@=                       ((@op2@>INT32_MAX) ? (@op1@ == 0 ? 0 : DBL_MAX * @op2@) : ldexp((double)@op1@,(int32_t)@op2@));',
             'bool'             => '@op3@=(@op2@ ? @op1@ * 2.0 : @op1@)'
         },
         'others'  => {
             # operand 2
             'int32_t'          => '@op3@=ldexp((double)@op1@,@op2@);',
             'int64_t'          => '@op3@=(@op2@<INT32_MIN) ? 0: ((@op2@>INT32_MAX) ? ((@op1@ > 0) ? DBL_MAX * @op2@ : ((@op1@ < 0) ? - DBL_MAX * @op2@ : 0)): ldexp((double)@op1@,(int32_t)@op2@));',
             'uint32_t'         => '@op3@=                       ((@op2@>INT32_MAX) ? ((@op1@ > 0) ? DBL_MAX * @op2@ : ((@op1@ < 0) ? - DBL_MAX * @op2@ : 0)): ldexp((double)@op1@,(int32_t)@op2@));',
             'uint64_t'         => '@op3@=                       ((@op2@>INT32_MAX) ? ((@op1@ > 0) ? DBL_MAX * @op2@ : ((@op1@ < 0) ? - DBL_MAX * @op2@ : 0)): ldexp((double)@op1@,(int32_t)@op2@));',
             'bool'             => '@op3@=(@op2@ ? @op1@ * 2.0 : @op1@)'
         },
         'bool'  => { # same as uint32_t
             # operand 2
             'int32_t'          => '@op3@=ldexp((double)@op1@,@op2@);',
             'int64_t'          => '@op3@=(@op2@<INT32_MIN) ? 0: ((@op2@>INT32_MAX) ? (@op1@ == 0 ? 0 : DBL_MAX * @op2@) : ldexp((double)@op1@,(int32_t)@op2@));',
             'uint32_t'         => '@op3@=                       ((@op2@>INT32_MAX) ? (@op1@ == 0 ? 0 : DBL_MAX * @op2@) : ldexp((double)@op1@,(int32_t)@op2@));',
             'uint64_t'         => '@op3@=                       ((@op2@>INT32_MAX) ? (@op1@ == 0 ? 0 : DBL_MAX * @op2@) : ldexp((double)@op1@,(int32_t)@op2@));',
             'bool'             => '@op3@=(@op2@ ? @op1@ * 2.0 : @op1@)'
         },
     },
},
'copysign'       => { 
     'float'  => '@op3@=copysign((double)@op1@,(double)@op2@);',
     'double' => '@op3@=copysign((double)@op1@,(double)@op2@);',
},
'absolute'=> { 
     'others'          => {
        'float'           => '@op2@ = fabsf(@op1@);',
        'double'          => '@op2@ = fabs(@op1@);',
        'float _Complex'  => '@op2@ = cabsf(@op1@);',
        'double _Complex' => '@op2@ = cabs(@op1@);',
        'int32_t'         => '@op2@ = abs(@op1@);',
        'int64_t'         => '@op2@ = labs(@op1@);',
        'uint32_t'        => '@op2@ = @op1@;',
        'uint64_t'        => '@op2@ = @op1@;',
        'bool'            => '@op2@ = @op1@;',
     },
},
'fabs'=> { 
     'others'          => {
        'float'           => '@op2@ = fabsf(@op1@);',
        'double'          => '@op2@ = fabs(@op1@);',
        'float _Complex'  => '@op2@ = cabsf(@op1@);',
        'double _Complex' => '@op2@ = cabs(@op1@);',
        'int32_t'         => '@op2@ = abs(@op1@);',
        'int64_t'         => '@op2@ = labs(@op1@);',
        'uint32_t'        => '@op2@ = @op1@;',
        'uint64_t'        => '@op2@ = @op1@;',
        'bool'            => '@op2@ = @op1@;',
     },
},
'signbit'=> { 
     'others'          => {
        'float'           => '@op2@ = (*(uint32_t*)(&(@op1@)))>>31;',
        'double'          => '@op2@ = (*(uint64_t*)(&(@op1@)))>>63;',
        'int32_t'         => '@op2@ = (@op1@>=0) ? 0 : 1;',
        'int64_t'         => '@op2@ = (@op1@>=0) ? 0 : 1;',
        'uint32_t'        => '@op2@ = 0;',
        'uint64_t'        => '@op2@ = 0;',
        'bool'            => '@op2@ = 0;',
     },
},
);

%irregular_intrinsic_table = (
'arcsin' => 'asin',      'arccos' => 'acos',      'arctan' => 'atan',
'arcsinh'=> 'asinh',     'arccosh'=> 'acosh',     'arctanh'=> 'atanh',
#'power'  => 'pow',
'rint'   => 'round',
'arctan2' => 'atan2',
'conj'=> { 
     'float'           => 'conjf',
     'float _Complex'  => 'conjf',
     'others'          => 'conj',
},
'conjugate'=> { 
     'float'           => 'conjf',
     'float _Complex'  => 'conjf',
     'others'          => 'conj',
},
#'signbit'=> { 
#     'others'          => 'signbit',
#},
'isfinite'=> { 
     'others'          => 'isfinite',
},
#'isinf'=> { 
#     'others'         => 'isinf',
#},
#'isnan'=> { 
#     'others'         => 'isnan',
#},
#'real'=> { 
#     'float'           => 'crealf',
#     'float _Complex'  => 'crealf',
#     'others'          => 'creal',
#},
#'imag'=> { 
#     'float'           => 'cimagf',
#     'float _Complex'  => 'cimagf',
#     'others'          => 'cimag',
#},
'angle'=> { 
     'float'           => 'cargf',
     'float _Complex'  => 'cargf',
     'others'          => 'carg',
},
);

%c_intrinsic_types_table = (
# name         in, out 
'abs'      => [$i32_t, $i32_t],
'labs'     => [$i64_t, $i64_t],
# math.h
#'cosf'    => [$f32_t, $f32_t],
#'cos'     => [$f64_t, $f64_t],
#'sinf'    => [$f32_t, $f32_t],
#'sin'     => [$f64_t, $f64_t],
#'tanf'    => [$f32_t, $f32_t],
#'tan'     => [$f64_t, $f64_t],
#'asinf'   => [$f32_t, $f32_t],
#'asin'    => [$f64_t, $f64_t],
#'acosf'   => [$f32_t, $f32_t],
#'acos'    => [$f64_t, $f64_t],
#'atanf'   => [$f32_t, $f32_t],
#'atan'    => [$f64_t, $f64_t],
'atan2f'  => [$f32_t, $f32_t, $f32_t],
'atan2'   => [$f64_t, $f64_t, $f64_t],
#'sinhf'   => [$f32_t, $f32_t],
#'sinh'    => [$f64_t, $f64_t],
#'coshf'   => [$f32_t, $f32_t],
#'cosh'    => [$f64_t, $f64_t],
#'tanhf'   => [$f32_t, $f32_t],
#'tanh'    => [$f64_t, $f64_t],
#'asinhf'  => [$f32_t, $f32_t],
#'asinh'   => [$f64_t, $f64_t],
#'acoshf'  => [$f32_t, $f32_t],
#'acosh'   => [$f64_t, $f64_t],
#'atanhf'  => [$f32_t, $f32_t],
#'atanh'   => [$f64_t, $f64_t],
#'expf'    => [$f32_t, $f32_t],
#'exp'     => [$f64_t, $f64_t],
'exp2f'   => [$f32_t, $f32_t],
'exp2'    => [$f64_t, $f64_t],
'expm1f'  => [$f32_t, $f32_t],
'expm1'   => [$f64_t, $f64_t],
'ldexpf'  => [$f32_t, $i32_t, $f32_t],
'ldexp'   => [$f64_t, $i32_t, $f64_t],
#'logf'    => [$f32_t, $f32_t],
#'log'     => [$f64_t, $f64_t],
'log2f'   => [$f32_t, $f32_t],
'log2'    => [$f64_t, $f64_t],
'log10f'  => [$f32_t, $f32_t],
'log10'   => [$f64_t, $f64_t],
'log1pf'  => [$f32_t, $f32_t],
'log1p'   => [$f64_t, $f64_t],
'sqrtf'   => [$f32_t, $f32_t],
'sqrt'    => [$f64_t, $f64_t],
'cbrtf'   => [$f32_t, $f32_t],
'cbrt'    => [$f64_t, $f64_t],
#'powf'    => [$f32_t, $f32_t],
#'pow'     => [$f64_t, $f64_t],
'hypotf'  => [$f32_t, $f32_t],
'hypot'   => [$f64_t, $f64_t],
'fabsf'   => [$f32_t, $f32_t],
'fabs'    => [$f64_t, $f64_t],
'erff'    => [$f32_t, $f32_t],
'erf'     => [$f64_t, $f64_t],
'lgammaf' => [$f32_t, $f32_t],
'lgamma'  => [$f64_t, $f64_t],
'tgammaf' => [$f32_t, $f32_t],
'tgamma'  => [$f64_t, $f64_t],
'ceilf'   => [$f32_t, $f32_t],
'ceil'    => [$f64_t, $f64_t],
'floorf'  => [$f32_t, $f32_t],
'floor'   => [$f64_t, $f64_t],
'rintf'   => [$f32_t, $f32_t],
'rint'    => [$f64_t, $f64_t],
#'lrintf'  => [$f32_t, $i64_t],
#'lrint'   => [$f64_t, $i64_t],
#'llrintf' => [$f32_t, $i64_t],
#'llrint'  => [$f64_t, $i64_t],
'roundf'  => [$f32_t, $f32_t],
'round'   => [$f64_t, $f64_t],
'truncf'  => [$f32_t, $f32_t],
'trunc'   => [$f64_t, $f64_t],
'fmodf'   => [$f32_t, $f32_t],
'fmod'    => [$f64_t, $f64_t],
#'remainderf'=> [$f32_t, $f32_t],
#'remainder' => [$f64_t, $f64_t],
#'remquof '=> [$f32_t, $f32_t],
#'remquo'  => [$f64_t, $f64_t],
'copysignf'=> [$f32_t, $f32_t, $f32_t],
'copysign' => [$f64_t, $f64_t, $f64_t],
#'nanf'    => [$f32_t, $f32_t],
#'nan'     => [$f64_t, $f64_t],
'nextafterf'=> [$f32_t, $f32_t, $f32_t],
'nextafter' => [$f64_t, $f64_t, $f64_t],
#'signbit'  => [$f64_t, $bool_t],
'isfinite' => [$f64_t, $bool_t],
'isinf'    => [$f64_t, $bool_t],
'isnan'    => [$f64_t, $bool_t],
'fmaxf'    => [$f32_t, $f32_t, $f32_t],
'fmax'     => [$f64_t, $f64_t, $f64_t],
'fminf'    => [$f32_t, $f32_t, $f32_t],
'fmin'     => [$f64_t, $f64_t, $f64_t],
#
# complex.h
#'ccosf'   => [$c64_t, $c64_t],
#'ccos'    => [$c128_t, $c128_t],
#'csinf'   => [$c64_t, $c64_t],
#'csin'    => [$c128_t, $c128_t],
#'ctanf'   => [$c64_t, $c64_t],
#'ctan'    => [$c128_t, $c128_t],
#'casinf'  => [$c64_t, $c64_t],
#'casin'   => [$c128_t, $c128_t],
#'cacosf'  => [$c64_t, $c64_t],
#'cacos'   => [$c128_t, $c128_t],
#'catanf'  => [$c64_t, $c64_t],
#'catan'   => [$c128_t, $c128_t],
#'csinhf'  => [$c64_t, $c64_t],
#'csinh'   => [$c128_t, $c128_t],
#'ccoshf'  => [$c64_t, $c64_t],
#'ccosh'   => [$c128_t, $c128_t],
#'ctanhf'  => [$c64_t, $c64_t],
#'ctanh'   => [$c128_t, $c128_t],
#'casinhf' => [$c64_t, $c64_t],
#'casinh'  => [$c128_t, $c128_t],
#'cacoshf' => [$c64_t, $c64_t],
#'cacosh'  => [$c128_t, $c128_t],
#'catanhf' => [$c64_t, $c64_t],
#'catanh'  => [$c128_t, $c128_t],
#'cexpf'   => [$c64_t, $c64_t],
#'cexp'    => [$c128_t, $c128_t],
#'clogf'   => [$c64_t, $c64_t],
#'clog'    => [$c128_t, $c128_t],
'cabsf'   => [$c64_t, $f32_t],
'cabs'    => [$c128_t, $f64_t],
'cpowf'   => [$c64_t, $c64_t],
'cpow'    => [$c128_t, $c128_t],
#'csqrtf'  => [$c64_t, $c64_t],
#'csqrt'   => [$c128_t, $c128_t],
'cargf'   => [$c64_t, $f32_t],
'carg'    => [$c128_t, $f64_t],
'cimagf'  => [$c64_t, $f32_t],
'cimag'   => [$c128_t, $f64_t],
'conjf'   => [$c64_t, $c64_t],
'conj'    => [$c128_t, $c128_t],
'cprojf'  => [$c64_t, $c64_t],
'cproj'   => [$c128_t, $c128_t],
'crealf'  => [$c64_t, $f32_t],
'creal'   => [$c128_t, $f64_t],
);
} 
