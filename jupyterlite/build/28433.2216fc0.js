"use strict";(self.webpackChunk_JUPYTERLAB_CORE_OUTPUT=self.webpackChunk_JUPYTERLAB_CORE_OUTPUT||[]).push([[28433],{28433:(t,a,n)=>{n.r(a),n.d(a,{troff:()=>h});var r={};function e(t){if(t.eatSpace())return null;var a=t.sol(),n=t.next();if("\\"===n)return t.match("fB")||t.match("fR")||t.match("fI")||t.match("u")||t.match("d")||t.match("%")||t.match("&")?"string":t.match("m[")?(t.skipTo("]"),t.next(),"string"):t.match("s+")||t.match("s-")?(t.eatWhile(/[\d-]/),"string"):t.match("(")||t.match("*(")?(t.eatWhile(/[\w-]/),"string"):"string";if(a&&("."===n||"'"===n)&&t.eat("\\")&&t.eat('"'))return t.skipToEnd(),"comment";if(a&&"."===n){if(t.match("B ")||t.match("I ")||t.match("R "))return"attribute";if(t.match("TH ")||t.match("SH ")||t.match("SS ")||t.match("HP "))return t.skipToEnd(),"quote";if(t.match(/[A-Z]/)&&t.match(/[A-Z]/)||t.match(/[a-z]/)&&t.match(/[a-z]/))return"attribute"}t.eatWhile(/[\w-]/);var e=t.current();return r.hasOwnProperty(e)?r[e]:null}function c(t,a){return(a.tokens[0]||e)(t,a)}const h={name:"troff",startState:function(){return{tokens:[]}},token:function(t,a){return c(t,a)}}}}]);
//# sourceMappingURL=28433.2216fc0.js.map