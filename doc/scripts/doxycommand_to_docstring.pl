#!/usr/bin/perl

use strict;
use warnings;
use utf8;
use feature ':5.10';


my $reach_command = 0;
my $reach_func = 0;
my $reach_li = 0;

my $pydoc_move = 0;
my $base_indent = "";

my $item_mode = 0;
sub FUNCTION {1};
sub CLASS {2};

my $pydoc = 0;
my $code_block  = 0; # @code block

my @briefs = ();
my @details = ();
my @params = ();
my @returns = ();
my @notes = ();
my @attentions = ();
my @warnings = ();
my @examples = ();
my @seealsoes = ();
#my @raises = ();
my @_mv_briefs = ();
my @_mv_details = ();
my @_mv_params = ();
my @_mv_returns = ();
my @_mv_notes = ();
my @_mv_attentions = ();
my @_mv_seealsoes = ();
my @_mv_examples = ();

my $buffering_status = 0;
sub BRIEFS {1};
sub PARAMS {2};
sub RETURNS {3};
sub DETAILS {4};
sub NOTES {5};
sub ATTENTIONS {6};
sub WARNINGS {7};
sub EXAMPLES {8};
sub SEEALSOES {9};
#sub RAISES {10};

my $LENGTH_MAX = 90;

my $rename_output = 0;
if(@ARGV == 2){
  $rename_output = 1; 
}

if($rename_output == 1){
  open(RENAMELIST, "+>>", $ARGV[1]);
}


my $src = $ARGV[0];
$src =~ s/.dox$//;
open(OUT, '>',$src);
open(SRC, $ARGV[0]);
# main exec
while (my $line = <SRC>){
 
  if(($reach_command == 1)&&($line =~/^\s*(def|cdef|cpdef|class|cdef class|cpdef class)/)){
    $line =~ /^(\s*)(def|cdef|cpdef|class|cdef class|cpdef class)/;
    $base_indent = $1 . "    ";
    $reach_func = 1;
  }
  
  if(($reach_func == 1)&&(!($line=~/\s*:/))){
    print OUT $line;
    next;
  }

  if(($reach_command == 1)
     &&($line !~ /^\s*#.*$/)
     #&&($line !~ /^\s*##.*$/)
     &&($reach_func == 0)){ 
    #$reach_command = 0;
    print OUT $line;
    next;
  }

  if(($line =~/^\s*## .*$/)||($line =~ /^\s*##$/)){
    #$line =~ /^(\s*)##/;
    #$base_indent = $1 . "    ";
    buffer_clear();  
    
    $item_mode = FUNCTION; 
    $reach_command = 1;
    
  }

  if($reach_command == 0){
    print OUT $line;
    next;
  }

  if($line =~/^\s*# \@pydoc\s*$/){
    $pydoc = 1;
  }
  if($line =~/^\s*# \@endpydoc\s*$/){
    $pydoc = 0;
  } 
  if(($line =~/^\s*# \@rename (.+)$/)&&($rename_output == 1)){
    print RENAMELIST $1. "\n";
  }
  if($line =~ /^\s*# \@pydocmove/){
    $pydoc_move = 1;
  }


  if($line =~/^\s*# \@(code|verbatim)/){
    $code_block = 1;
      $line =~ s/\@$1/ \@n/g;
  }
  if($line =~/^\s*# \@(endcode|endverbatim)/){
    $code_block = 0;
  }

  if(($item_mode != 0)&&($pydoc == 1 )){

    if($line =~/^\s*# \@brief (.+)$/){
      #push(@briefs, $1);
      my $tmp = $1;
      $briefs[0] = dot_spacing($tmp);
      $buffering_status = BRIEFS;
      next;
    }

    if($line =~/^\s*# \@details (.+)$/){
      #push(@details, $1);
      my $tmp = $1;
      $details[0] = dot_spacing($tmp);
      $buffering_status = DETAILS;
      next;
    }
    if($line =~/^\s*# \@details\s*$/){
      push(@details, "");
      $buffering_status = DETAILS;
      next;
    }


    if($line =~/^\s*# \@param (.+)$/){
      push(@params, $1);
      $params[$#params] = $params[$#params] . "##";
      $buffering_status = PARAMS;
      next;
    }
    
    if(($line =~/^\s*# \@return (.+)$/) || ($line =~ /^\s*# \@retval (.+)$/)){
      push(@returns, $1);
      $returns[$#returns] = $returns[$#returns] ."##";
      $buffering_status = RETURNS;
      next;
    }
    
    if($line =~/^\s*# \@note (.+)$/){
      #push(@notes, $1);
      my $tmp = $1;
      $notes[0] = dot_spacing($tmp);
      $buffering_status = NOTES;
      next;
    }
    if($line =~/^\s*# \@note\s*$/){
      $notes[0] = "";
      $buffering_status = NOTES;
      next;
    }

    if($line =~/\@raise (.+)$/){
      push(@attentions, $1);
      $buffering_status = ATTENTIONS;
      next;
    }

    if($line =~/^\s*# \@warning (.+)$/){
      push(@warnings, "WARINING ".$1);
      $buffering_status = WARNINGS;
      next;
    }

    if($line =~/\s*# \@par (.+)$/){
      my $par = $1;
      if($par =~ /Example/){
        push(@examples, "");
        $buffering_status = EXAMPLES;
      }else{
        push(@details, ($par . "\@n"));
        $buffering_status = DETAILS;
        #$buffering_status = 0;
      }
      next;
    }

    if($line =~ /\@sa (.+)$/) {
      my $tmp = $1;
      if($tmp =~ /(\S+?) : (.+?) "(\S+?)"/){
	$tmp = $3 . " : " . $2; 
      }
      push(@seealsoes, $tmp);
      $buffering_status = SEEALSOES;
      next;
    }

    if($line =~ /# \@sa\s*$/){
      $buffering_status = SEEALSOES;
      next;
    }

    if(($buffering_status == 9)&&($line =~ /\@li \@ref\s+(.+)$/)){
      my $tmp = $1;
      $tmp =~ s/\\s+/\\s/g;
      if($tmp =~ /(\S+?)\s*"(\S+?)"\s*:\s*(.+)/){
        $tmp = $1 . " : " . $3;
      }
      push(@seealsoes, $tmp);
      next;
    }

    $line =~ s/\@li \@ref (\S+?)\s*"(\S+?)"\s*(.+)/- \`$2\` $3/;
    $line =~ s/\@li (.+)$/- $1/;

    if($line =~ /^\s*# ((?!\@).*)$/){
      my $pattern = $1;
      if($pattern !~ /./){
	$buffering_status = 0;
	next;
      }
      # Remove leading space except for code blocks.
      if($code_block == 0) {
        $pattern =~ s/^\s*//g;
      }else{
        $pattern = "\$\$" . $pattern . "\@n";
      }
      
      given($buffering_status){
	when (1) { reform_rectangle($pattern, \@briefs); next; }
	when (2) { reform_rectangle_sub($pattern, \@params); next; }
	when (3) { reform_rectangle_sub($pattern, \@returns); next; }
	when (4) { reform_rectangle($pattern, \@details); next; }
	when (5) { reform_rectangle($pattern, \@notes); next; }
	when (6) { $attentions[$#attentions] = $attentions[$#attentions] . "##" . $pattern; next; }
	when (7) { push(@warnings, $pattern); next; } #Unused
	when (8) { reform_rectangle($pattern ,\@examples); next; }
	when (9) { $seealsoes[$#seealsoes] = $seealsoes[$#seealsoes] . "##" . $pattern; next; }
	default {next;}
      }
    }
  }

  if((@briefs) #&& ($line =~/^(\s*)(def|cdef|cpdef|class|cdef class|cpdef class) .+:/)){ 
     && ($reach_func == 1)){
 
    print OUT $line;
    $briefs[0] = "\"\"\"" . $briefs[0];

    if($pydoc_move == 1){
      @_mv_briefs = @briefs;
      @_mv_details = @details;
      @_mv_params = @params;
      @_mv_returns = @returns;
      @_mv_notes = @notes;
      @_mv_attentions = @attentions;
      @_mv_seealsoes = @seealsoes;
      @_mv_examples = @examples;
    }else{
      dox_move(*OUT, $base_indent, \@briefs, \@details, \@params, \@returns, \@attentions, \@notes, \@seealsoes, \@examples);
    }

    buffer_clear();
    $reach_func = 0;
    $reach_command = 0;
    next;
  }elsif((!@briefs)&&($reach_func == 1)){
    print OUT $line;
    $reach_func = 0;
    $reach_command = 0;
  }

  #if($line !~/^\s*[#|##]/){
  #  print $line;
  #}
}

if($rename_output == 1){
  close(RENAMELIST);
}
close(SRC);
close(OUT);

if(@_mv_briefs){
  my @_indata = ();
  open(_fh, '<', $src);
  @_indata = <_fh>;
  close(_fh);

  open(_fh, '>', $src);
  foreach(@_indata){
    if(($_ =~ /(\s*)# \@pydocmove/)&&(@_mv_briefs)){
      my $indent = $1 . "    ";
      dox_move(*_fh , $indent, \@_mv_briefs, \@_mv_details, \@_mv_params, \@_mv_returns, \@_mv_attentions, \@_mv_notes, \@_mv_seealsoes, \@_mv_examples);
      @_mv_briefs = ();
      @_mv_params = ();
      @_mv_returns = ();
      @_mv_details = ();
      @_mv_notes = ();
      @_mv_attentions = ();
      @_mv_examples = ();
      @_mv_seealsoes = ();
      next;
    }
    print _fh $_;
  }
  close(_fh);
}

sub insert_multiline_sub {
  local *FILE = $_[0];
  my $base_indent = $_[1];
  my $add_indent = $_[2];
  my $reform = $_[3];
  my $params = $_[4];
  foreach my $p (@$params){
     my @strs = split(/##/, $p);
      while(my ($i,$s) = each @strs){
        my $_indent;

        if($i == 0){
          if(length($s) == 0){
            next;
          }
          $_indent = $base_indent;# . "    ";
        }else{
          $_indent = $base_indent . "    ";# . "    ";
        }

        my $_base_string = pop_newline($s);
        if($reform==0){ # Example skip
          print FILE $_indent, $_base_string , "\n";
          next;
        }
        
        if((length($_indent . $_base_string) < $LENGTH_MAX)
           ||($_base_string =~ /\|.+\|.+\|.+\|/)
           ||($_base_string =~ /^\$\$/)
        ){
          $_base_string =~ s/\s*\$\$//;
          if(length($_base_string)==0){
            next;
          }
          print FILE $_indent, $_base_string , "\n";
        }else{
          $_base_string =~ s/\$\$//;
          my @letters = split(/\s/, $_base_string);
          my $buf = "";
          foreach my $l (@letters){
            if($buf eq ""){
              $buf = $l;
            }else{
              if(length($_indent . $buf . " " . $l) < $LENGTH_MAX){
                $buf = $buf . " " . $l;
              }else{
                print FILE $_indent, $buf , "\n";
                $buf = $l;
              }
            }
          }
          if($buf ne ""){
            if($add_indent == 1){
              print FILE $_indent, "    " ,$buf , "\n";
            }else{
              print FILE $_indent, $buf , "\n";
            }
          }
        }
      }
  }
}

sub init_docparams {
  @briefs = ();
  @params = ();
  @returns = ();
  @details = ();
  @notes = ();
  @attentions = ();
  @warnings = ();
  @examples = ();
  @seealsoes = ();
}

sub init_readstatus{
  $item_mode = 0;
  $buffering_status = 0;
  $code_block = 0;
  $pydoc = 0;
  $pydoc_move = 0;
}

sub buffer_clear{
  init_docparams();    

  init_readstatus();
}

sub pop_newline{
  my $str = shift;

  $str =~ s/\@n$//;

  $str =~ s/\@f\[/..math::/g;
  $str =~ s/\@f\]//g;
  $str =~ s/\@f\$(.+?)\@f\$/:math:\`$1\`/g;

  $str =~ s/\\([a-zA-Z_])/\\\\$1/g;

  $str =~ s/<[^\']*?>//g;

  $str =~ s/\@ref (\S+?) "(\S+?)"/\`$2\`/g;
  $str =~ s/\@.+? //g;
  $str =~ s/%(.+?)([\.\s])/$1$2/g;

  $str =~ s/\s+$//;  

  return $str
}

sub dot_spacing{
  my $str = shift;
  $str =~ s/([\S])$/$1 /;

  return $str;
}

sub reform_rectangle{
  my($pattern, $add_list) = @_;

  if((@$add_list[$#$add_list] =~ /^\-/)&&($pattern =~ /^\-/)){ #@li
    push(@$add_list, "");
  }

  @$add_list[$#$add_list] = @$add_list[$#$add_list] . dot_spacing($pattern);

  if($pattern =~ /\@n/){
     push(@$add_list, "");
  }
}

sub reform_rectangle_sub{
  my($pattern, $add_list) = @_;

  if($pattern =~ /^\-/){ #@li
    @$add_list[$#$add_list] = @$add_list[$#$add_list] . "##";
  }

  if($pattern =~/\|.+\|.+\|.+\|/){ #markdown table
    push(@$add_list, "##" . $pattern . "##");
    return;
  }

  @$add_list[$#$add_list] = @$add_list[$#$add_list] . dot_spacing($pattern);
  if($pattern =~ /\@n/){
     @$add_list[$#$add_list] = @$add_list[$#$add_list] . "##";
  }
}

sub dox_move{
  local *FILE = $_[0];
  my $base_indent = $_[1];
  my $briefs = $_[2];
  my $details = $_[3];
  my $params  = $_[4];
  my $returns  = $_[5];
  my $attentions  = $_[6];
  my $notes  = $_[7];
  my $seealsoes  = $_[8];
  my $examples = $_[9];

    if(@$briefs){
      insert_multiline_sub(*FILE, $base_indent, 0, 1, $briefs);
      print FILE "\n";
    }   
 
    if((@$details)&&(@$details[0] =~ /\S/)){
      insert_multiline_sub(*FILE, $base_indent, 0, 1, $details);
      print FILE "\n";
    }
    
    if(@$params){
      print FILE $base_indent, "Args:\n";
      insert_multiline_sub(*FILE, $base_indent . "    ", 0, 1, $params);
      print FILE "\n";
    }

    if(@$returns){
      print FILE $base_indent, "Returns:\n";
      insert_multiline_sub(*FILE, $base_indent . "    ", 0, 1, $returns);
      print FILE "\n";
    }
    
    if(@$attentions){
      print FILE $base_indent, "Raises:\n";
      insert_multiline_sub(*FILE, $base_indent . "    ", 1, 1, $attentions);
      print FILE "\n";
    }

    if(@$notes){
      print FILE $base_indent, "Note:\n";
      insert_multiline_sub(*FILE, $base_indent . "    ", 0, 1, $notes);
      print FILE "\n";
    }
    
    if(@$seealsoes){
      print FILE $base_indent, "See Also:\n";
      insert_multiline_sub(*FILE, $base_indent . "    ", 1, 1, $seealsoes);
      print FILE "\n";
    }
    
    if(@$examples){
      print FILE $base_indent, "Examples:\n";
      insert_multiline_sub(*FILE, $base_indent . "    ", 0, 1, $examples);
      print FILE "\n";
    }
    print FILE $base_indent, "\"\"\"", "\n";
}
